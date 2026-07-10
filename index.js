'use strict'
require('dotenv').config()
const path=require('path')
const crypto=require('crypto')
const express=require('express')
const cors=require('cors')
const mongoose=require('mongoose')
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

const PSMODEL_ENDPOINT=process.env.PSMODEL_ENDPOINT||'https://api.deepseek.com/chat/completions'
const PSMODEL_API_KEY=process.env.PSMODEL_API_KEY
const PSMODEL_MODEL=process.env.PSMODEL_MODEL||'deepseek-chat'
const PSMODEL_TIMEOUT_MS=parseInt(process.env.PSMODEL_TIMEOUT_MS||'60000',10)
const PSMODEL_TEMPERATURE=parseFloat(process.env.PSMODEL_TEMPERATURE||'0.7')

const PSMODELCHATHISDB_URI=process.env.PSMODELCHATHISDB_URI

const EMBEDDING_MODEL_NAME=process.env.EMBEDDING_MODEL_NAME||'BAAI/bge-base-en-v1.5'
const EMBEDDING_CACHE_DIR=process.env.EMBEDDING_CACHE_DIR||path.join(process.cwd(),'.fastembed_cache')

const question_limit=parseInt(process.env.QUESTION_LIMIT||'25',10)
const QUESTION_BANK_TOP_K=parseInt(process.env.QUESTION_BANK_TOP_K||'12',10)
const KNOWLEDGE_BASE_TOP_K=parseInt(process.env.KNOWLEDGE_BASE_TOP_K||'10',10)
const GENERATION_BATCH_SIZE=parseInt(process.env.GENERATION_BATCH_SIZE||'10',10)
const SAVE_GENERATED_TO_QDRANT=process.env.SAVE_GENERATED_TO_QDRANT!=='false'
const QDRANT_UPSERT_BATCH_SIZE=parseInt(process.env.QDRANT_UPSERT_BATCH_SIZE||'64',10)

const qdrant=new QdrantClient({url:QDRANT_URL,apiKey:QDRANT_API_KEY})

const chatHistorySchema=new mongoose.Schema({
requestId:{type:String,index:true},
adminQuery:String,
examType:String,
subject:String,
topic:String,
chapter:String,
keywords:[String],
difficulty:String,
requestedCount:Number,
generatedCount:Number,
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
const system='You are an intent extraction engine for a Civil Services exam question generation system. Extract structured parameters from the admin natural language request. Always respond with strict JSON only, no markdown, no prose, no code fences.'
const user=`Admin request: "${query}"

Return ONLY a JSON object in this exact shape:
{"count":null,"examType":null,"subject":null,"topic":null,"chapter":null,"keywords":[],"difficulty":null}

Rules:
count is the integer number of questions requested, or null if not mentioned.
examType is the exam name and stage if mentioned, for example "UPSC Prelims", "BPSC", "State PSC Mains", or null.
topic is the specific topic the questions should be about.
chapter is the book chapter or syllabus section if identifiable, otherwise same as topic or null.
subject is the broader subject area such as Polity, History, Geography, Economy, Science, Environment or Current Affairs, inferred from the topic if not explicit.
keywords is an array of related search terms derived from the request.
difficulty is "Easy", "Moderate" or "Difficult" if mentioned or implied, otherwise null.`
return {system,user}
}

async function analyzeQuery(query){
const {system,user}=buildAnalyzePrompt(query)
const content=await callPSModel(system,user)
const cleaned=extractJsonBlock(content,'{','}')
try{
const parsed=JSON.parse(cleaned)
return {
count:Number.isFinite(parsed.count)?parseInt(parsed.count,10):null,
examType:parsed.examType||null,
subject:parsed.subject||null,
topic:parsed.topic||null,
chapter:parsed.chapter||null,
keywords:Array.isArray(parsed.keywords)?parsed.keywords:[],
difficulty:parsed.difficulty||null
}
}catch(e){
return {count:null,examType:null,subject:null,topic:null,chapter:null,keywords:[],difficulty:null}
}
}

function buildPrompt({examType,topic,subject,difficulty,batchCount,pyqText,kbText}){
const exam=examType||'Civil Services'
const system=`You are a senior question setter for ${exam} examinations with years of experience designing previous year papers. You generate fresh, original multiple choice questions. You never copy or lightly reword previous year questions. You use the supplied previous year questions only to learn the examiner's style, difficulty, wording pattern and framing. You use the supplied knowledge base context only as the factual source for the new questions. You always respond with strict JSON only, no markdown, no prose, no code fences.`
const user=`Topic: ${topic}
Subject: ${subject||'General Studies'}
Exam: ${exam}
Difficulty: ${difficulty||'Moderate, matching the exam standard'}
Generate exactly ${batchCount} new original MCQs.

Previous year questions for style, pattern and difficulty reference only:
${pyqText}

Knowledge base context to use as the factual source for new questions:
${kbText}

Return ONLY a JSON array with exactly ${batchCount} objects in this exact shape, and nothing else:
[{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"correct_answer":"A","explanation":"...","difficulty":"Easy|Moderate|Difficult","topic":"${topic}","subject":"${subject||''}"}]`
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

async function streamPSModel(system,user,onToken){
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
max_tokens:4000,
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

function parseQuestionsJSON(raw){
if(!raw) return []
const cleaned=extractJsonBlock(raw,'[',']')
let parsed
try{
parsed=JSON.parse(cleaned)
}catch(e){
return []
}
if(!Array.isArray(parsed)) return []
return parsed.filter(q=>q&&typeof q.question==='string'&&q.options&&typeof q.options==='object')
}

async function generateBatchStreaming(params,onToken){
for(let attempt=0;attempt<2;attempt++){
const {system,user}=buildPrompt(params)
const content=await streamPSModel(system,user,onToken)
const questions=parseQuestionsJSON(content)
if(questions.length) return questions
}
return []
}

async function saveGeneratedQuestions(questions,meta){
if(!SAVE_GENERATED_TO_QDRANT||!questions.length) return 0
const texts=questions.map(q=>[q.question,...Object.values(q.options||{})].join(' '))
const vectors=await embedTexts(texts)
const points=questions.map((q,i)=>({
id:crypto.randomUUID(),
vector:vectors[i],
payload:{
exam:meta.examType||null,
subject:q.subject||meta.subject||null,
topic:q.topic||meta.topic,
chapter:meta.chapter||null,
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
mongoError:lastMongoError
})
}catch(e){
res.status(500).json({ok:false,error:e.message})
}
})

app.post('/api/questions/generate',requireAdmin,async(req,res)=>{
const body=req.body||{}
const query=(body.query||'').trim()
if(!query&&!(body.topic||'').trim()){
return res.status(400).json({error:'query or topic is required'})
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
const analyzed=query?await analyzeQuery(query):{count:null,examType:null,subject:null,topic:null,chapter:null,keywords:[],difficulty:null}

const topic=((body.topic||analyzed.topic)||'').trim()
if(!topic){
sendEvent('error',{error:'Could not determine a topic from the request, please rephrase or include a topic explicitly'})
return res.end()
}
const examType=((body.examType||body.exam||analyzed.examType)||'').trim()
const subject=((body.subject||analyzed.subject)||'').trim()
const chapter=((body.chapter||analyzed.chapter)||'').trim()
const difficulty=((body.difficulty||analyzed.difficulty)||'').trim()
const keywords=body.keywords||analyzed.keywords||[]

let count=parseInt(body.count,10)
if(!Number.isFinite(count)||count<=0){
count=Number.isFinite(analyzed.count)&&analyzed.count>0?analyzed.count:10
}
const requestedCount=count
count=clamp(count,1,question_limit)
const limited=requestedCount>question_limit
const limitMessage=limited?`Try to generate questions below ${question_limit}.`:null

const searchText=buildSearchText({topic,examType,subject,chapter,keywords})
const queryVector=await embedOne(searchText)

const [pyqPoints,kbPoints]=await Promise.all([
searchQuestionBank(queryVector,QUESTION_BANK_TOP_K),
searchKnowledgeBase(queryVector,KNOWLEDGE_BASE_TOP_K)
])

const pyqText=formatPYQs(pyqPoints)
const kbText=formatKnowledge(kbPoints)

sendEvent('meta',{
requestId,
adminQuery:query||null,
topic,
examType:examType||null,
subject:subject||null,
chapter:chapter||null,
difficulty:difficulty||null,
requestedCount,
questionLimit:question_limit,
limitedToQuestionLimit:limited,
limitMessage,
pyqReferencesUsed:pyqPoints.length,
knowledgeChunksUsed:kbPoints.length
})

const batches=[]
let remaining=count
while(remaining>0){
const size=Math.min(GENERATION_BATCH_SIZE,remaining)
batches.push(size)
remaining-=size
}

let questions=[]
for(let b=0;b<batches.length;b++){
const batchCount=batches[b]
sendEvent('batch_start',{batch:b+1,totalBatches:batches.length,count:batchCount})
const params={examType,topic,subject,chapter,difficulty,batchCount,pyqText,kbText}
const batchQuestions=await generateBatchStreaming(params,delta=>{
sendEvent('token',{batch:b+1,content:delta})
})
questions=questions.concat(batchQuestions)
sendEvent('batch_done',{batch:b+1,totalBatches:batches.length,questions:batchQuestions})
}

questions=questions.slice(0,count)

let savedToQdrant=0
try{
savedToQdrant=await saveGeneratedQuestions(questions,{examType,subject,topic,chapter,difficulty,requestId})
}catch(e){
console.error('[saveGeneratedQuestions]',e.message)
}

let mongoId=null
try{
if(await connectMongo()){
const doc=await ChatHistory.create({
requestId,
adminQuery:query||null,
examType:examType||null,
subject:subject||null,
topic,
chapter:chapter||null,
keywords,
difficulty:difficulty||null,
requestedCount,
generatedCount:questions.length,
limitedToQuestionLimit:limited,
questionLimit:question_limit,
pyqReferencesUsed:pyqPoints.length,
knowledgeChunksUsed:kbPoints.length,
questions,
model:PSMODEL_MODEL,
savedToQdrant
})
mongoId=doc._id.toString()
}
}catch(e){
console.error('[mongo save]',e.message)
}

sendEvent('done',{
requestId,
mongoId,
generatedCount:questions.length,
savedToQdrant,
questions
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
