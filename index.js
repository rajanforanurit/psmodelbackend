'use strict'
require('dotenv').config()
const path=require('path')
const crypto=require('crypto')
const express=require('express')
const cors=require('cors')
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

const EMBEDDING_MODEL_NAME=process.env.EMBEDDING_MODEL_NAME||'BAAI/bge-base-en-v1.5'
const EMBEDDING_CACHE_DIR=process.env.EMBEDDING_CACHE_DIR||path.join(process.cwd(),'.fastembed_cache')

const MAX_QUESTIONS_PER_REQUEST=100
const QUESTION_BANK_TOP_K=parseInt(process.env.QUESTION_BANK_TOP_K||'12',10)
const KNOWLEDGE_BASE_TOP_K=parseInt(process.env.KNOWLEDGE_BASE_TOP_K||'10',10)
const GENERATION_BATCH_SIZE=parseInt(process.env.GENERATION_BATCH_SIZE||'10',10)
const GENERATION_CONCURRENCY=parseInt(process.env.GENERATION_CONCURRENCY||'3',10)
const SAVE_GENERATED_TO_QDRANT=process.env.SAVE_GENERATED_TO_QDRANT!=='false'
const QDRANT_UPSERT_BATCH_SIZE=parseInt(process.env.QDRANT_UPSERT_BATCH_SIZE||'64',10)

const qdrant=new QdrantClient({url:QDRANT_URL,apiKey:QDRANT_API_KEY})

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

function withTimeout(promise,ms,label){
let timer
const timeout=new Promise((_,reject)=>{
timer=setTimeout(()=>reject(new Error(`${label||'operation'} timed out`)),ms)
})
return Promise.race([promise,timeout]).finally(()=>clearTimeout(timer))
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
temperature:PSMODEL_TEMPERATURE,
max_tokens:4000,
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

function parseQuestionsJSON(raw){
if(!raw) return []
let cleaned=raw.trim()
cleaned=cleaned.replace(/^```json/i,'').replace(/^```/,'').replace(/```$/,'').trim()
const start=cleaned.indexOf('[')
const end=cleaned.lastIndexOf(']')
if(start!==-1&&end!==-1&&end>start) cleaned=cleaned.slice(start,end+1)
let parsed
try{
parsed=JSON.parse(cleaned)
}catch(e){
return []
}
if(!Array.isArray(parsed)) return []
return parsed.filter(q=>q&&typeof q.question==='string'&&q.options&&typeof q.options==='object')
}

async function generateBatch(params){
for(let attempt=0;attempt<2;attempt++){
const {system,user}=buildPrompt(params)
const content=await callPSModel(system,user)
const questions=parseQuestionsJSON(content)
if(questions.length) return questions
}
return []
}

async function runPool(items,concurrency,worker){
const results=new Array(items.length)
let cursor=0
async function next(){
while(cursor<items.length){
const current=cursor++
results[current]=await worker(items[current],current)
}
}
const workers=Array.from({length:Math.min(concurrency,items.length)},()=>next())
await Promise.all(workers)
return results
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

app.get('/',(req,res)=>{
res.json({ok:true,service:'psmodel-question-generator'})
})

app.get('/health',async(req,res)=>{
try{
const collections=await qdrant.getCollections()
res.json({ok:true,time:new Date().toISOString(),collections:collections.collections.map(c=>c.name)})
}catch(e){
res.status(500).json({ok:false,error:e.message})
}
})

app.post('/api/questions/generate',requireAdmin,async(req,res)=>{
try{
const body=req.body||{}
const topic=(body.topic||'').trim()
if(!topic) return res.status(400).json({error:'topic is required'})
const examType=(body.examType||body.exam||'').trim()
const subject=(body.subject||'').trim()
const chapter=(body.chapter||'').trim()
const difficulty=(body.difficulty||'').trim()
const keywords=body.keywords

let count=parseInt(body.count,10)
if(!Number.isFinite(count)||count<=0) count=10
const requestedCount=count
count=clamp(count,1,MAX_QUESTIONS_PER_REQUEST)
const limited=requestedCount>MAX_QUESTIONS_PER_REQUEST

const searchText=buildSearchText({topic,examType,subject,chapter,keywords})
const queryVector=await embedOne(searchText)

const [pyqPoints,kbPoints]=await Promise.all([
searchQuestionBank(queryVector,QUESTION_BANK_TOP_K),
searchKnowledgeBase(queryVector,KNOWLEDGE_BASE_TOP_K)
])

const pyqText=formatPYQs(pyqPoints)
const kbText=formatKnowledge(kbPoints)

const batches=[]
let remaining=count
while(remaining>0){
const size=Math.min(GENERATION_BATCH_SIZE,remaining)
batches.push(size)
remaining-=size
}

const requestId=crypto.randomUUID()
const batchResults=await withTimeout(
runPool(batches,GENERATION_CONCURRENCY,batchCount=>generateBatch({examType,topic,subject,chapter,difficulty,batchCount,pyqText,kbText})),
PSMODEL_TIMEOUT_MS*Math.ceil(batches.length/GENERATION_CONCURRENCY)+10000,
'question generation'
)

let questions=batchResults.flat()
questions=questions.slice(0,count)

let saved=0
try{
saved=await saveGeneratedQuestions(questions,{examType,subject,topic,chapter,difficulty,requestId})
}catch(e){
console.error('[saveGeneratedQuestions]',e.message)
}

res.json({
requestId,
topic,
examType:examType||null,
subject:subject||null,
chapter:chapter||null,
requestedCount,
limitedTo100:limited,
generatedCount:questions.length,
pyqReferencesUsed:pyqPoints.length,
knowledgeChunksUsed:kbPoints.length,
savedToQdrant:saved,
questions
})
}catch(e){
console.error('[generate]',e)
if(!res.headersSent) res.status(500).json({error:e.message||'Internal error'})
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

app.use((req,res)=>{
res.status(404).json({error:'Not found'})
})

app.use((err,req,res,next)=>{
console.error('[global error]',err)
if(!res.headersSent) res.status(500).json({error:'Unexpected error'})
})

app.listen(PORT,()=>{
console.log(`PSMODEL question generation backend running on port ${PORT}`)
})

module.exports=app
