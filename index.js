'use strict'
require('dotenv').config()
const path=require('path')
const crypto=require('crypto')
const express=require('express')
const cors=require('cors')
const mongoose=require('mongoose')
const PDFDocument=require('pdfkit')
const {QdrantClient}=require('@qdrant/js-client-rest')
const {EmbeddingModel,FlagEmbedding}=require('fastembed')

const app=express()
const PORT=process.env.PORT||8080
const ALLOWED_ORIGINS=(process.env.ALLOWED_ORIGINS||'*').split(',').map(s=>s.trim()).filter(Boolean)
const corsOpts={
origin:(origin,cb)=>{
if(!origin||ALLOWED_ORIGINS.includes('*')||ALLOWED_ORIGINS.includes(origin)) return cb(null,true)
cb(new Error('Not allowed by CORS'))
},
methods:['GET','POST','OPTIONS'],
allowedHeaders:['Content-Type','Authorization','x-admin-key']
}
app.use(cors(corsOpts))
app.options('*',cors(corsOpts))
app.use(express.json({limit:'2mb'}))

const QDRANT_URL=process.env.QDRANT_URL
const QDRANT_API_KEY=process.env.QDRANT_API_KEY
const QDRANT_QUESTION_BANK_COLLECTION=process.env.QDRANT_QUESTION_BANK_COLLECTION||'question_bank'
const QDRANT_KNOWLEDGE_BASE_COLLECTION=process.env.QDRANT_KNOWLEDGE_BASE_COLLECTION||'knowledge_base'
const QDRANT_GENERATED_QUESTIONS_COLLECTION=process.env.QDRANT_GENERATED_QUESTIONS_COLLECTION||'generated_questions'

const ADMIN_API_KEY=process.env.ADMIN_API_KEY

const PSMODEL_ENDPOINT=process.env.PSMODEL_ENDPOINT
const PSMODEL_API_KEY=process.env.PSMODEL_API_KEY
const PSMODEL_MODEL=process.env.PSMODEL_MODEL
const PSMODEL_TIMEOUT_MS=parseInt(process.env.PSMODEL_TIMEOUT_MS||'60000',10)
const PSMODEL_TEMPERATURE=parseFloat(process.env.PSMODEL_TEMPERATURE||'0.7')

const PSMODELCHATHISDB_URI=process.env.PSMODELCHATHISDB_URI

const EMBEDDING_MODEL_NAME=process.env.EMBEDDING_MODEL_NAME||'BAAI/bge-base-en-v1.5'
const EMBEDDING_CACHE_DIR=process.env.EMBEDDING_CACHE_DIR||path.join(process.cwd(),'.fastembed_cache')

// question_limit is now a TOTAL cap across every topic in a single request (not per-topic).
const question_limit=parseInt(process.env.QUESTION_LIMIT||'100',10)
// Max number of distinct topics/subjects allowed in one request, to keep latency/cost sane.
const MAX_TOPICS=parseInt(process.env.MAX_TOPICS||'8',10)
const QUESTION_BANK_TOP_K=parseInt(process.env.QUESTION_BANK_TOP_K||'12',10)
const KNOWLEDGE_BASE_TOP_K=parseInt(process.env.KNOWLEDGE_BASE_TOP_K||'10',10)
// Batches are NDJSON (one question per line) with per-line parsing and shortfall-only retries,
// so a broken/truncated line no longer costs you the whole batch. That makes larger batches safe.
const GENERATION_BATCH_SIZE=parseInt(process.env.GENERATION_BATCH_SIZE||'25',10)
// How many topics run concurrently in a multi-topic request. Independent topics don't need to
// wait on each other; keep this modest to stay under your PSMODEL provider's rate limits.
const TOPIC_CONCURRENCY=parseInt(process.env.TOPIC_CONCURRENCY||'2',10)
const SAVE_GENERATED_TO_QDRANT=process.env.SAVE_GENERATED_TO_QDRANT!=='false'
const QDRANT_UPSERT_BATCH_SIZE=parseInt(process.env.QDRANT_UPSERT_BATCH_SIZE||'64',10)

const qdrant=new QdrantClient({url:QDRANT_URL,apiKey:QDRANT_API_KEY})

const chatHistorySchema=new mongoose.Schema({
requestId:{type:String,index:true},
adminQuery:String,
// Deprecated single-topic fields, kept populated (from the first topic) for backward compatibility.
examType:String,
subject:String,
topic:String,
chapter:String,
keywords:[String],
difficulty:String,
// New multi-topic breakdown.
topics:[mongoose.Schema.Types.Mixed],
requestedCount:Number,
generatedCount:Number,
// True if one or more topics/batches stopped early or failed, so generatedCount < requestedCount.
partial:{type:Boolean,default:false},
limitedToQuestionLimit:Boolean,
questionLimit:Number,
pyqReferencesUsed:Number,
knowledgeChunksUsed:Number,
questions:[mongoose.Schema.Types.Mixed],
model:String,
savedToQdrant:Number
},{timestamps:true})

const ChatHistory=mongoose.models.ChatHistory||mongoose.model('ChatHistory',chatHistorySchema,'psmodel_chat_history')

let mongoConnectPromise=null
let lastMongoError=null

mongoose.connection.on('connected',()=>{
console.log('[mongoose] connected')
lastMongoError=null
})
mongoose.connection.on('error',e=>{
console.error('[mongoose] connection error',e.message)
lastMongoError=e.message
})
mongoose.connection.on('disconnected',()=>{
console.log('[mongoose] disconnected')
})

function connectMongo(){
if(!PSMODELCHATHISDB_URI) return Promise.resolve(false)
if(mongoose.connection.readyState===1) return Promise.resolve(true)
if(!mongoConnectPromise){
mongoConnectPromise=mongoose.connect(PSMODELCHATHISDB_URI,{serverSelectionTimeoutMS:8000})
.then(()=>{
lastMongoError=null
return true
})
.catch(e=>{
console.error('[mongo connect]',e.message)
lastMongoError=e.message
mongoConnectPromise=null
return false
})
}
return mongoConnectPromise
}

let embedderPromise=null
function getEmbedder(){
if(!embedderPromise){
embedderPromise=FlagEmbedding.init({
model:EmbeddingModel.BGEBaseENV15,
cacheDir:EMBEDDING_CACHE_DIR,
maxLength:512
})
}
return embedderPromise
}

async function embedTexts(texts){
if(!texts||!texts.length) return []
const embedder=await getEmbedder()
const out=[]
for await(const batch of embedder.embed(texts,32)){
for(const vec of batch) out.push(Array.from(vec))
}
return out
}

async function embedOne(text){
const vectors=await embedTexts([text])
return vectors[0]
}

function requireAdmin(req,res,next){
if(!ADMIN_API_KEY) return res.status(500).json({error:'Server not configured with ADMIN_API_KEY'})
const headerKey=req.headers['x-admin-key']
const bearer=(req.headers.authorization||'').replace(/^Bearer\s+/i,'')
const key=headerKey||bearer
if(!key||key!==ADMIN_API_KEY) return res.status(401).json({error:'Unauthorized'})
next()
}

function clamp(n,min,max){
return Math.max(min,Math.min(max,n))
}

function sanitizeFileName(name){
const base=(name||'questions').toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'').slice(0,60)
return base||'questions'
}

function buildSearchText({topic,examType,subject,chapter,keywords}){
const kw=Array.isArray(keywords)?keywords.join(' '):(keywords||'')
return [examType,subject,chapter,topic,kw].filter(Boolean).join(' ').trim()
}

async function searchQuestionBank(vector,topK){
const result=await qdrant.query(QDRANT_QUESTION_BANK_COLLECTION,{
query:vector,
limit:topK,
with_payload:true
})
return result.points||[]
}

async function searchKnowledgeBase(vector,topK){
const result=await qdrant.query(QDRANT_KNOWLEDGE_BASE_COLLECTION,{
query:vector,
limit:topK,
with_payload:true
})
return result.points||[]
}

function formatPYQs(points){
if(!points.length) return 'None found'
return points.map((p,i)=>{
const pl=p.payload||{}
const opts=pl.options?Object.entries(pl.options).map(([k,v])=>`${k}) ${v}`).join(' | '):''
const lines=[`${i+1}. [${pl.exam||'Exam'}${pl.year?' '+pl.year:''}] ${pl.question||''}`]
if(opts) lines.push(`Options: ${opts}`)
if(pl.answer) lines.push(`Answer: ${pl.answer}`)
return lines.join('\n')
}).join('\n\n')
}

function formatKnowledge(points){
if(!points.length) return 'None found'
return points.map((p,i)=>{
const pl=p.payload||{}
const label=pl.chapter||pl.topic||pl.subject||pl.source||'Reference'
return `[${i+1}] (${label}) ${pl.text||''}`
}).join('\n\n')
}

function extractJsonBlock(raw,openChar,closeChar){
let cleaned=(raw||'').trim()
cleaned=cleaned.replace(/^```json/i,'').replace(/^```/,'').replace(/```$/,'').trim()
const start=cleaned.indexOf(openChar)
const end=cleaned.lastIndexOf(closeChar)
if(start!==-1&&end!==-1&&end>start) cleaned=cleaned.slice(start,end+1)
return cleaned
}

function buildAnalyzePrompt(query){
const system='You are an intent extraction engine for a Civil Services exam question generation system. Extract structured parameters from the admin natural language request. The request may cover ONE topic or MULTIPLE distinct topics/subjects/difficulty levels in the same message. Always respond with strict JSON only, no markdown, no prose, no code fences.'
const user=`Admin request: "${query}"

Return ONLY a JSON object in this exact shape:
{"requests":[{"count":null,"examType":null,"subject":null,"topic":null,"chapter":null,"keywords":[],"difficulty":null}]}

Rules:
Create ONE object per distinct topic/subject/difficulty combination the admin asked for. If only one topic is mentioned, return an array with exactly one object.
count is the integer number of questions requested for that specific topic, or null if not mentioned or if only a single combined total was given for multiple topics.
examType is the exam name and stage if mentioned, for example "UPSC Prelims", "BPSC", "State PSC Mains", or null. If mentioned once for the whole request, repeat it on every object.
topic is the specific topic those questions should be about.
chapter is the book chapter or syllabus section if identifiable, otherwise same as topic or null.
subject is the broader subject area such as Polity, History, Geography, Economy, Science, Environment or Current Affairs, inferred from the topic if not explicit.
keywords is an array of related search terms derived from the request for that topic.
difficulty is "Easy", "Moderate" or "Difficult" if mentioned or implied for that topic, otherwise null.
Never merge two clearly different topics into one object.`
return {system,user}
}

function normalizeSpec(raw){
return {
count:Number.isFinite(raw.count)?parseInt(raw.count,10):null,
examType:raw.examType||null,
subject:raw.subject||null,
topic:raw.topic||null,
chapter:raw.chapter||null,
keywords:Array.isArray(raw.keywords)?raw.keywords:[],
difficulty:raw.difficulty||null
}
}

async function analyzeQuery(query){
const {system,user}=buildAnalyzePrompt(query)
const content=await callPSModel(system,user)
const cleaned=extractJsonBlock(content,'{','}')
try{
const parsed=JSON.parse(cleaned)
let requests=Array.isArray(parsed.requests)?parsed.requests:null
if(!requests){
// Backward-compatible fallback in case the model returns the old flat shape.
if(parsed.topic||parsed.subject) requests=[parsed]
}
if(!requests||!requests.length) return []
return requests.map(normalizeSpec).filter(s=>s.topic)
}catch(e){
return []
}
}

function applyFallbacks(spec,fallback){
return {
count:spec.count,
examType:spec.examType||fallback.examType||null,
subject:spec.subject||fallback.subject||null,
topic:spec.topic,
chapter:spec.chapter||fallback.chapter||null,
keywords:(spec.keywords&&spec.keywords.length)?spec.keywords:(fallback.keywords||[]),
difficulty:spec.difficulty||fallback.difficulty||null
}
}

// Resolve how many questions each topic should get before the overall limit is applied.
function resolveSpecCounts(specs,bodyCount){
const anySpecHasCount=specs.some(s=>Number.isFinite(s.count)&&s.count>0)
if(specs.length===1){
const c=Number.isFinite(bodyCount)&&bodyCount>0?bodyCount:(Number.isFinite(specs[0].count)&&specs[0].count>0?specs[0].count:10)
specs[0].requestedCount=c
return specs
}
if(anySpecHasCount){
specs.forEach(s=>{
s.requestedCount=Number.isFinite(s.count)&&s.count>0?s.count:10
})
return specs
}
const total=Number.isFinite(bodyCount)&&bodyCount>0?bodyCount:specs.length*10
const base=Math.floor(total/specs.length)
let remainder=total-base*specs.length
specs.forEach(s=>{
s.requestedCount=base+(remainder>0?1:0)
if(remainder>0) remainder--
})
return specs
}

// Apply the overall QUESTION_LIMIT across all topics combined, scaling proportionally if needed.
function applyOverallLimit(specs,limit){
const totalRequested=specs.reduce((a,s)=>a+s.requestedCount,0)
if(totalRequested<=limit){
specs.forEach(s=>{s.count=s.requestedCount})
return {limited:false,totalRequested,totalCount:totalRequested}
}
const scaled=specs.map(s=>Math.max(1,Math.floor(s.requestedCount*limit/totalRequested)))
let sum=scaled.reduce((a,b)=>a+b,0)
let diff=limit-sum
let i=0
while(diff!==0&&specs.length>0&&i<10000){
const idx=i%specs.length
if(diff>0){scaled[idx]++;diff--}
else if(scaled[idx]>1){scaled[idx]--;diff++}
i++
}
specs.forEach((s,idx)=>{s.count=scaled[idx]})
return {limited:true,totalRequested,totalCount:scaled.reduce((a,b)=>a+b,0)}
}

// NDJSON instead of a JSON array: one object per line. This means a truncated stream or one
// malformed line only costs that line, not the entire batch — the accumulator below parses
// and accepts lines as they complete, independent of whatever comes after them.
function buildPrompt({examType,topic,subject,difficulty,batchCount,pyqText,kbText,avoidList}){
const exam=examType||'Civil Services'
const system=`You are a senior ${exam} question setter. Write fresh, original MCQs — never copy or lightly reword the sample previous-year questions below; use them only to match style, tone and difficulty. Use the knowledge base text as the sole factual source. Write each explanation as a direct, self-contained statement of fact — never open with a meta-phrase like "As per the knowledge," "Based on the provided context," or "According to the source." Output NDJSON only: exactly one valid JSON object per line, no surrounding array brackets, no commas between lines, no blank lines, no markdown, no code fences, no numbering, no text before or after the lines.`

const avoidBlock=(avoidList&&avoidList.length)?`\nDo not repeat or closely rephrase any of these already-used question stems:\n${avoidList.map(s=>`- ${s}`).join('\n')}\n`:''

const user=`Topic: ${topic}
Subject: ${subject||'General Studies'}
Exam: ${exam}
Difficulty: ${difficulty||'Moderate, matching the exam standard'}
Generate exactly ${batchCount} new original MCQs. Output exactly ${batchCount} lines, each a standalone JSON object in this shape:
{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"correct_answer":"A","explanation":"...","difficulty":"Easy|Moderate|Difficult","topic":"${topic}","subject":"${subject||''}"}
${avoidBlock}
Previous year questions (style/pattern/difficulty reference only, do not copy):
${pyqText}

Knowledge base context (factual source for new questions):
${kbText}`
return {system,user}
}

async function callPSModel(system,user){
const controller=new AbortController()
const timer=setTimeout(()=>controller.abort(),PSMODEL_TIMEOUT_MS)
try{
const response=await fetch(PSMODEL_ENDPOINT,{
method:'POST',
headers:{
'Content-Type':'application/json',
Authorization:`Bearer ${PSMODEL_API_KEY}`
},
body:JSON.stringify({
model:PSMODEL_MODEL,
messages:[
{role:'system',content:system},
{role:'user',content:user}
],
temperature:0,
max_tokens:800,
stream:false
}),
signal:controller.signal
})
if(!response.ok){
const errText=await response.text().catch(()=>'')
throw new Error(`PSMODEL request failed with status ${response.status}: ${errText}`)
}
const data=await response.json()
return data?.choices?.[0]?.message?.content||''
}finally{
clearTimeout(timer)
}
}

async function streamPSModel(system,user,onToken,maxTokens){
const controller=new AbortController()
const timer=setTimeout(()=>controller.abort(),PSMODEL_TIMEOUT_MS)
let full=''
try{
const response=await fetch(PSMODEL_ENDPOINT,{
method:'POST',
headers:{
'Content-Type':'application/json',
Authorization:`Bearer ${PSMODEL_API_KEY}`
},
body:JSON.stringify({
model:PSMODEL_MODEL,
messages:[
{role:'system',content:system},
{role:'user',content:user}
],
temperature:PSMODEL_TEMPERATURE,
max_tokens:maxTokens||4000,
stream:true
}),
signal:controller.signal
})
if(!response.ok||!response.body){
const errText=await response.text().catch(()=>'')
throw new Error(`PSMODEL request failed with status ${response.status}: ${errText}`)
}
const reader=response.body.getReader()
const decoder=new TextDecoder('utf-8')
let buffer=''
while(true){
const {done,value}=await reader.read()
if(done) break
buffer+=decoder.decode(value,{stream:true})
let sepIndex
while((sepIndex=buffer.indexOf('\n\n'))!==-1){
const rawEvent=buffer.slice(0,sepIndex)
buffer=buffer.slice(sepIndex+2)
const lines=rawEvent.split('\n')
for(const line of lines){
const trimmed=line.trim()
if(!trimmed.startsWith('data:')) continue
const payload=trimmed.slice(5).trim()
if(payload==='[DONE]') continue
try{
const json=JSON.parse(payload)
const delta=json?.choices?.[0]?.delta?.content
if(delta){
full+=delta
if(onToken) onToken(delta)
}
}catch(e){}
}
}
}
return full
}finally{
clearTimeout(timer)
}
}

// Parses a single NDJSON line into a question object, tolerating a stray trailing comma or
// accidental code-fence/array-bracket noise the model might still slip in.
function parseQuestionLine(line){
let cleaned=(line||'').trim()
if(!cleaned) return null
cleaned=cleaned.replace(/^```json/i,'').replace(/^```/,'').replace(/```$/,'').trim()
cleaned=cleaned.replace(/,\s*$/,'')
if(!cleaned.startsWith('{')){
const s=cleaned.indexOf('{')
const e=cleaned.lastIndexOf('}')
if(s===-1||e===-1||e<=s) return null
cleaned=cleaned.slice(s,e+1)
}
try{
const obj=JSON.parse(cleaned)
if(obj&&typeof obj.question==='string'&&obj.question.trim()&&obj.options&&typeof obj.options==='object') return obj
return null
}catch(e){
return null
}
}

// Normalizes a question stem for duplicate detection: lowercase, strip punctuation, collapse
// whitespace. Doesn't need to be perfect — it only needs to catch near-identical repeats.
function normalizeStem(text){
return (text||'').toLowerCase().replace(/[^a-z0-9\s]/g,'').replace(/\s+/g,' ').trim().slice(0,120)
}

// Buffers streamed token deltas and emits complete lines to onLine as soon as a newline
// arrives, so questions can be accepted (and even shown to the user) mid-stream instead of
// only after the whole batch finishes.
function createLineAccumulator(onLine){
let buffer=''
return {
push(delta){
buffer+=delta
let idx
while((idx=buffer.indexOf('\n'))!==-1){
const line=buffer.slice(0,idx)
buffer=buffer.slice(idx+1)
if(line.trim()) onLine(line)
}
},
flush(){
if(buffer.trim()) onLine(buffer)
buffer=''
}
}
}

// Generates exactly params.batchCount NEW (non-duplicate) questions for one batch, retrying
// only the shortfall (not the whole batch) up to 2 extra times if lines were dropped, invalid,
// or duplicates. dedupState is shared across the whole request (all topics, all batches).
async function generateQuestionsForBatch(params,dedupState,onQuestion){
let stillNeeded=params.batchCount
let collected=[]
for(let attempt=0;attempt<3&&stillNeeded>0;attempt++){
const avoidList=dedupState.recentTexts.slice(-20)
const passParams={...params,batchCount:stillNeeded,avoidList}
const maxTokens=clamp(stillNeeded*230+300,600,16000)
const {system,user}=buildPrompt(passParams)
const acc=createLineAccumulator(line=>{
const q=parseQuestionLine(line)
if(!q) return
const stem=normalizeStem(q.question)
if(dedupState.seen.has(stem)) return
dedupState.seen.add(stem)
dedupState.recentTexts.push(q.question.slice(0,140))
collected.push(q)
if(onQuestion) onQuestion(q)
})
try{
await streamPSModel(system,user,delta=>acc.push(delta),maxTokens)
}catch(e){
// fall through to flush whatever was collected before the error, then retry the shortfall
}
acc.flush()
stillNeeded=params.batchCount-collected.length
}
return collected
}

// Runs an array through `worker` with at most `limit` in flight at once, preserving output
// order. Used to process several topics concurrently without unbounded parallelism.
//
// IMPORTANT: each worker call is individually try/caught. Previously a single rejected worker
// (e.g. topic 3's embedding/Qdrant lookup failing) would reject the whole Promise.all below,
// which meant every other topic's already-generated questions - even ones fully finished and
// already streamed to the client - were discarded and the request ended in a bare error. Now a
// failing topic is captured as a normal (non-throwing) result so the rest of the request still
// completes and whatever was generated is still returned.
async function runWithConcurrency(items,limit,worker){
const results=new Array(items.length)
let cursor=0
async function runner(){
while(true){
const i=cursor++
if(i>=items.length) return
try{
results[i]=await worker(items[i],i)
}catch(e){
results[i]={
spec:items[i],
questions:[],
pyqReferencesUsed:0,
knowledgeChunksUsed:0,
failed:true,
stoppedEarly:true,
stopReason:(e&&e.message)||'Unexpected error while generating this topic'
}
}
}
}
const runners=Array.from({length:Math.max(1,Math.min(limit,items.length))},()=>runner())
await Promise.all(runners)
return results
}

// Generates all questions for a single topic spec, in batches of GENERATION_BATCH_SIZE,
// emitting SSE progress events along the way. dedupState is shared across the whole request
// so duplicate stems get caught even across different topics/batches.
async function generateForTopic(spec,sendEvent,topicIndex,totalTopics,dedupState){
// Setup (embedding + Qdrant lookups) used to be unguarded: if it threw, the exception
// propagated out of this whole function, which - before runWithConcurrency was hardened -
// wiped out every other topic's results too. It's now caught here as well, defensively, so
// this topic just reports itself as failed instead of generating anything.
let pyqPoints,kbPoints
try{
const searchText=buildSearchText(spec)
const queryVector=await embedOne(searchText)
;[pyqPoints,kbPoints]=await Promise.all([
searchQuestionBank(queryVector,QUESTION_BANK_TOP_K),
searchKnowledgeBase(queryVector,KNOWLEDGE_BASE_TOP_K)
])
}catch(e){
const stopReason=`Could not look up reference material for "${spec.topic}": ${(e&&e.message)||'unknown error'}`
sendEvent('topic_error',{topicIndex,totalTopics,topic:spec.topic,error:stopReason})
sendEvent('topic_done',{
topicIndex,totalTopics,topic:spec.topic,generatedCount:0,
totalBatches:0,completedBatches:0,stoppedEarly:true,stopReason,failed:true
})
return {spec,questions:[],pyqReferencesUsed:0,knowledgeChunksUsed:0,failed:true,stoppedEarly:true,stopReason}
}

const pyqText=formatPYQs(pyqPoints)
const kbText=formatKnowledge(kbPoints)

sendEvent('topic_start',{
topicIndex,
totalTopics,
topic:spec.topic,
subject:spec.subject,
examType:spec.examType,
difficulty:spec.difficulty,
requestedCount:spec.requestedCount,
count:spec.count,
pyqReferencesUsed:pyqPoints.length,
knowledgeChunksUsed:kbPoints.length
})

const batches=[]
let remaining=spec.count
while(remaining>0){
const size=Math.min(GENERATION_BATCH_SIZE,remaining)
batches.push(size)
remaining-=size
}

let questions=[]
let completedBatches=0
let stoppedEarly=false
let stopReason=null

for(let b=0;b<batches.length;b++){
const batchCount=batches[b]
sendEvent('batch_start',{topicIndex,totalTopics,topic:spec.topic,batch:b+1,totalBatches:batches.length,count:batchCount})
const params={examType:spec.examType,topic:spec.topic,subject:spec.subject,chapter:spec.chapter,difficulty:spec.difficulty,batchCount,pyqText,kbText}

let batchQuestions=[]
try{
batchQuestions=await generateQuestionsForBatch(params,dedupState,q=>{
sendEvent('question_ready',{topicIndex,totalTopics,topic:spec.topic,batch:b+1,question:q})
})
}catch(e){
// generateQuestionsForBatch already swallows its own retry errors and returns [] rather
// than throwing, but guard here too so one bad batch can never take the whole topic down.
batchQuestions=[]
}

sendEvent('batch_done',{topicIndex,totalTopics,topic:spec.topic,batch:b+1,totalBatches:batches.length,delivered:batchQuestions.length,requested:batchCount})

// A batch that produced NOTHING after all internal retries means something is
// systemically wrong (rate limit, outage, bad key) rather than a one-off shortfall.
// Stop here and hand back everything already generated in prior batches, instead of
// either losing it (old behavior on a hard crash) or silently grinding through more
// batches that are likely to fail the same way.
if(batchQuestions.length===0){
stoppedEarly=true
stopReason=`Batch ${b+1} of ${batches.length} produced no questions after retries, so the remaining batches for this topic were skipped.`
break
}

questions=questions.concat(batchQuestions)
completedBatches++
}

questions=questions.slice(0,spec.count).map(q=>({
...q,
topic:q.topic||spec.topic,
subject:q.subject||spec.subject||null
}))

sendEvent('topic_done',{
topicIndex,
totalTopics,
topic:spec.topic,
generatedCount:questions.length,
totalBatches:batches.length,
completedBatches,
stoppedEarly,
stopReason
})

return {spec,questions,pyqReferencesUsed:pyqPoints.length,knowledgeChunksUsed:kbPoints.length,stoppedEarly,stopReason}
}

async function saveGeneratedQuestions(questions,meta){
if(!SAVE_GENERATED_TO_QDRANT||!questions.length) return 0
const texts=questions.map(q=>[q.question,...Object.values(q.options||{})].join(' '))
const vectors=await embedTexts(texts)
const points=questions.map((q,i)=>({
id:crypto.randomUUID(),
vector:vectors[i],
payload:{
exam:q.examType||meta.examType||null,
subject:q.subject||meta.subject||null,
topic:q.topic||meta.topic,
chapter:q.chapter||meta.chapter||null,
question:q.question,
options:q.options,
correct_answer:q.correct_answer||null,
explanation:q.explanation||null,
difficulty:q.difficulty||meta.difficulty||null,
source:'generated',
generated_at:new Date().toISOString(),
request_id:meta.requestId
}
}))
for(let i=0;i<points.length;i+=QDRANT_UPSERT_BATCH_SIZE){
const batch=points.slice(i,i+QDRANT_UPSERT_BATCH_SIZE)
await qdrant.upsert(QDRANT_GENERATED_QUESTIONS_COLLECTION,{wait:true,points:batch})
}
return points.length
}

async function deleteGeneratedQuestionsByRequestId(requestId){
if(!requestId) return 0
const result=await qdrant.delete(QDRANT_GENERATED_QUESTIONS_COLLECTION,{
filter:{must:[{key:'request_id',match:{value:requestId}}]},
wait:true
})
return result
}

function streamQuestionsPDF(res,questions,meta){
const fileName=`psmodel_${sanitizeFileName(meta.topic)}.pdf`
res.setHeader('Content-Type','application/pdf')
res.setHeader('Content-Disposition',`attachment; filename="${fileName}"`)
const doc=new PDFDocument({margin:50,size:'A4'})
doc.pipe(res)
doc.fontSize(18).fillColor('#000000').text(meta.topic?`Question Set: ${meta.topic}`:'Generated Questions')
doc.moveDown(0.4)
const metaLine=[meta.examType,meta.subject,meta.difficulty].filter(Boolean).join('   |   ')
if(metaLine){
doc.fontSize(10).fillColor('#555555').text(metaLine)
doc.fillColor('#000000')
}
doc.moveDown()
const multiTopic=meta.multiTopic
questions.forEach((q,i)=>{
if(doc.y>700) doc.addPage()
doc.fontSize(12).fillColor('#000000').text(`${i+1}. ${q.question||''}`)
if(multiTopic&&(q.topic||q.subject)){
doc.fontSize(9).fillColor('#777777').text(`   ${[q.subject,q.topic].filter(Boolean).join(' / ')}`)
doc.fillColor('#000000')
}
doc.moveDown(0.2)
const opts=q.options||{}
Object.keys(opts).sort().forEach(k=>{
doc.fontSize(11).text(`   ${k}) ${opts[k]}`)
})
doc.moveDown(0.2)
if(q.correct_answer){
doc.fontSize(11).fillColor('#0a7d32').text(`   Correct Answer: ${q.correct_answer}`)
doc.fillColor('#000000')
}
if(q.explanation){
doc.fontSize(10).fillColor('#444444').text(`   Explanation: ${q.explanation}`)
doc.fillColor('#000000')
}
doc.moveDown()
})
doc.end()
}

app.get('/',(req,res)=>{
res.json({ok:true,service:'psmodel-question-generator'})
})

app.get('/health',async(req,res)=>{
try{
const collections=await qdrant.getCollections()
const mongoConnected=await connectMongo()
res.json({
ok:true,
time:new Date().toISOString(),
collections:collections.collections.map(c=>c.name),
mongoConfigured:!!PSMODELCHATHISDB_URI,
mongo:mongoConnected?'connected':'disconnected',
mongoError:lastMongoError,
questionLimit:question_limit,
maxTopics:MAX_TOPICS,
generationBatchSize:GENERATION_BATCH_SIZE,
topicConcurrency:TOPIC_CONCURRENCY
})
}catch(e){
res.status(500).json({ok:false,error:e.message})
}
})

app.post('/api/questions/generate',requireAdmin,async(req,res)=>{
const body=req.body||{}
const query=(body.query||'').trim()

// Explicit per-topic overrides from the client take priority over NL parsing.
const explicitTopics=Array.isArray(body.topics)?body.topics.map(normalizeSpec).filter(s=>s.topic):null

if(!query&&!(body.topic||'').trim()&&!(explicitTopics&&explicitTopics.length)){
return res.status(400).json({error:'query, topic, or topics is required'})
}

res.setHeader('Content-Type','text/event-stream')
res.setHeader('Cache-Control','no-cache')
res.setHeader('Connection','keep-alive')
res.setHeader('X-Accel-Buffering','no')
res.flushHeaders()

function sendEvent(event,data){
res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
}

const requestId=crypto.randomUUID()

try{
let specs=explicitTopics&&explicitTopics.length?explicitTopics:null

if(!specs){
if(query){
specs=await analyzeQuery(query)
}
if((!specs||!specs.length)&&(body.topic||'').trim()){
specs=[normalizeSpec({
topic:(body.topic||'').trim(),
examType:body.examType||body.exam||null,
subject:body.subject||null,
chapter:body.chapter||null,
keywords:body.keywords||[],
difficulty:body.difficulty||null,
count:Number.isFinite(parseInt(body.count,10))?parseInt(body.count,10):null
})]
}
}

if(!specs||!specs.length){
sendEvent('error',{error:'Could not determine a topic from the request, please rephrase or include a topic explicitly'})
return res.end()
}

if(specs.length>MAX_TOPICS){
specs=specs.slice(0,MAX_TOPICS)
}

const globalFallback={
examType:(body.examType||body.exam||'').trim()||null,
subject:(body.subject||'').trim()||null,
chapter:(body.chapter||'').trim()||null,
difficulty:(body.difficulty||'').trim()||null,
keywords:body.keywords||[]
}
specs=specs.map(s=>applyFallbacks(s,globalFallback))

const bodyCount=parseInt(body.count,10)
specs=resolveSpecCounts(specs,Number.isFinite(bodyCount)?bodyCount:null)

const {limited,totalRequested,totalCount}=applyOverallLimit(specs,question_limit)
const limitMessage=limited?`Total questions requested (${totalRequested}) exceeded the limit of ${question_limit} and were scaled down proportionally across topics.`:null

sendEvent('meta',{
requestId,
adminQuery:query||null,
topics:specs.map(s=>({
topic:s.topic,
examType:s.examType,
subject:s.subject,
chapter:s.chapter,
difficulty:s.difficulty,
requestedCount:s.requestedCount,
count:s.count
})),
totalTopics:specs.length,
questionLimit:question_limit,
requestedCount:totalRequested,
limitedToQuestionLimit:limited,
limitMessage
})

// Shared across every topic/batch in this request so duplicate stems get caught even when
// they show up under a different topic (common when topics overlap, e.g. two Polity subtopics).
const dedupState={seen:new Set(),recentTexts:[]}
const results=await runWithConcurrency(
specs,
TOPIC_CONCURRENCY,
(spec,i)=>generateForTopic(spec,sendEvent,i+1,specs.length,dedupState)
)

let questions=results.flatMap(r=>r.questions||[])
const pyqReferencesUsed=results.reduce((a,r)=>a+(r.pyqReferencesUsed||0),0)
const knowledgeChunksUsed=results.reduce((a,r)=>a+(r.knowledgeChunksUsed||0),0)

// Per-topic status so the client can show exactly what finished, what was partial, and why -
// instead of an all-or-nothing success/error.
const topicsResult=specs.map((s,i)=>{
const r=results[i]||{}
return {
topic:s.topic,
examType:s.examType,
subject:s.subject,
chapter:s.chapter,
difficulty:s.difficulty,
keywords:s.keywords,
requestedCount:s.requestedCount,
count:s.count,
generatedCount:(r.questions||[]).length,
stoppedEarly:!!r.stoppedEarly,
stopReason:r.stopReason||null,
failed:!!r.failed
}
})
const partial=topicsResult.some(t=>t.stoppedEarly||t.failed)

sendEvent('done',{
requestId,
generatedCount:questions.length,
totalTopics:specs.length,
partial,
topics:topicsResult,
questions
})

let savedToQdrant=0
try{
savedToQdrant=await saveGeneratedQuestions(questions,{requestId,examType:specs[0].examType,subject:specs[0].subject,topic:specs[0].topic,chapter:specs[0].chapter,difficulty:specs[0].difficulty})
}catch(e){
console.error('[saveGeneratedQuestions]',e.message)
}

let mongoId=null
try{
if(await connectMongo()){
const first=specs[0]
const doc=await ChatHistory.create({
requestId,
adminQuery:query||null,
examType:first.examType||null,
subject:first.subject||null,
topic:first.topic,
chapter:first.chapter||null,
keywords:first.keywords||[],
difficulty:first.difficulty||null,
topics:topicsResult,
requestedCount:totalRequested,
generatedCount:questions.length,
partial,
limitedToQuestionLimit:limited,
questionLimit:question_limit,
pyqReferencesUsed,
knowledgeChunksUsed,
questions,
model:PSMODEL_MODEL,
savedToQdrant
})
mongoId=doc._id.toString()
}
}catch(e){
console.error('[mongo save]',e.message)
}

sendEvent('persisted',{
requestId,
mongoId,
savedToQdrant
})
res.end()
}catch(e){
console.error('[generate]',e)
try{
sendEvent('error',{error:e.message||'Internal error'})
}catch(_){}
res.end()
}
})

app.post('/api/questions/list',requireAdmin,async(req,res)=>{
try{
const body=req.body||{}
const limit=clamp(parseInt(body.limit,10)||20,1,100)
const offset=body.offset||undefined
const result=await qdrant.scroll(QDRANT_GENERATED_QUESTIONS_COLLECTION,{
limit,
offset,
with_payload:true,
with_vector:false
})
res.json({points:result.points,nextOffset:result.next_page_offset||null})
}catch(e){
res.status(500).json({error:e.message||'Internal error'})
}
})

app.post('/api/questions/pdf',requireAdmin,async(req,res)=>{
try{
const body=req.body||{}
let questions=Array.isArray(body.questions)?body.questions:null
let meta={topic:body.topic||null,examType:body.examType||null,subject:body.subject||null,difficulty:body.difficulty||null}

if(!questions&&body.requestId){
if(await connectMongo()){
const doc=await ChatHistory.findOne({requestId:body.requestId}).lean()
if(doc){
questions=doc.questions||[]
const topicList=Array.isArray(doc.topics)?doc.topics.map(t=>t.topic).filter(Boolean):[]
meta={
topic:topicList.length>1?topicList.join(', '):doc.topic,
examType:doc.examType,
subject:topicList.length>1?null:doc.subject,
difficulty:topicList.length>1?null:doc.difficulty,
multiTopic:topicList.length>1
}
}
}
if(!questions||!questions.length){
const result=await qdrant.scroll(QDRANT_GENERATED_QUESTIONS_COLLECTION,{
filter:{must:[{key:'request_id',match:{value:body.requestId}}]},
limit:500,
with_payload:true,
with_vector:false
})
const points=result.points||[]
if(points.length){
questions=points.map(p=>({
question:p.payload.question,
options:p.payload.options,
correct_answer:p.payload.correct_answer,
explanation:p.payload.explanation,
difficulty:p.payload.difficulty,
topic:p.payload.topic,
subject:p.payload.subject
}))
const distinctTopics=[...new Set(points.map(p=>p.payload.topic).filter(Boolean))]
meta.topic=meta.topic||(distinctTopics.length>1?distinctTopics.join(', '):distinctTopics[0])
meta.examType=meta.examType||points[0].payload.exam
meta.subject=meta.subject||(distinctTopics.length>1?null:points[0].payload.subject)
meta.difficulty=meta.difficulty||(distinctTopics.length>1?null:points[0].payload.difficulty)
meta.multiTopic=meta.multiTopic||distinctTopics.length>1
}
}
}

if(!questions||!questions.length) return res.status(404).json({error:'No questions found to export'})
streamQuestionsPDF(res,questions,meta)
}catch(e){
if(!res.headersSent) res.status(500).json({error:e.message||'Internal error'})
else res.end()
}
})

app.post('/api/chat-history/list',requireAdmin,async(req,res)=>{
try{
if(!(await connectMongo())) return res.status(503).json({error:'MongoDB not configured or unavailable'})
const body=req.body||{}
const limit=clamp(parseInt(body.limit,10)||20,1,100)
const docs=await ChatHistory.find({}).sort({createdAt:-1}).limit(limit).lean()
res.json({items:docs})
}catch(e){
res.status(500).json({error:e.message||'Internal error'})
}
})

app.post('/api/chat-history/delete',requireAdmin,async(req,res)=>{
try{
if(!(await connectMongo())) return res.status(503).json({error:'MongoDB not configured or unavailable'})
const body=req.body||{}
const id=(body.id||'').trim()
if(!id) return res.status(400).json({error:'id is required'})
const doc=await ChatHistory.findById(id)
if(!doc) return res.status(404).json({error:'History item not found'})
let qdrantDeleted=false
try{
await deleteGeneratedQuestionsByRequestId(doc.requestId)
qdrantDeleted=true
}catch(e){
console.error('[qdrant delete]',e.message)
}
await ChatHistory.deleteOne({_id:id})
res.json({deleted:true,id,requestId:doc.requestId,qdrantDeleted})
}catch(e){
res.status(500).json({error:e.message||'Internal error'})
}
})

app.use((req,res)=>{
res.status(404).json({error:'Not found'})
})

app.use((err,req,res,next)=>{
console.error('[global error]',err)
if(!res.headersSent) res.status(500).json({error:'Unexpected error'})
})

app.listen(PORT,()=>{
console.log(`PSMODEL question generation backend running on port ${PORT}`)
connectMongo().then(ok=>{
console.log(ok?'[mongoose] initial connection succeeded':`[mongoose] initial connection failed: ${lastMongoError||'PSMODELCHATHISDB_URI not set'}`)
})
})

module.exports=app
