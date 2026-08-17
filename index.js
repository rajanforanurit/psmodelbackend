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

process.on('unhandledRejection',(reason)=>{
console.error('[unhandledRejection] backend kept alive despite this:',reason)
})
process.on('uncaughtException',(err)=>{
console.error('[uncaughtException] backend kept alive despite this:',err)
})
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
// Bumped default vs. the original 60s: bilingual (English+Hindi) output, plus
// long statement-based questions for UPSC/PCS topics, roughly doubles the
// tokens per question, so generation legitimately takes longer.
const PSMODEL_TIMEOUT_MS=parseInt(process.env.PSMODEL_TIMEOUT_MS||'120000',10)
const PSMODEL_TEMPERATURE=parseFloat(process.env.PSMODEL_TEMPERATURE||'0.7')

const PSMODELCHATHISDB_URI=process.env.PSMODELCHATHISDB_URI

// Cluster for the "predicted questions" collection the admin panel can push
// reviewed/generated questions into. Separate from PSMODELCHATHISDB_URI,
// which only stores generation chat history/audit records.
const PREDICTQUES_URI=process.env.PREDICTQUES_URI

// The predicted-questions schema stores correct_answer as a Number, while the
// model still answers with an option letter (A/B/C/D) internally. This is the
// index the letter "A" maps to. Default 0 (A=0,B=1,C=2,D=3). Flip to '1' via
// env if your frontend expects 1-based indices instead.
const PREDICTED_ANSWER_INDEX_BASE=parseInt(process.env.PREDICTED_ANSWER_INDEX_BASE||'0',10)

const EMBEDDING_MODEL_NAME=process.env.EMBEDDING_MODEL_NAME||'BAAI/bge-base-en-v1.5'
const EMBEDDING_CACHE_DIR=process.env.EMBEDDING_CACHE_DIR||path.join(process.cwd(),'.fastembed_cache')

const question_limit=parseInt(process.env.QUESTION_LIMIT||'100',10)
const MAX_TOPICS=parseInt(process.env.MAX_TOPICS||'8',10)
const QUESTION_BANK_TOP_K=parseInt(process.env.QUESTION_BANK_TOP_K||'12',10)
const KNOWLEDGE_BASE_TOP_K=parseInt(process.env.KNOWLEDGE_BASE_TOP_K||'10',10)
const GENERATION_BATCH_SIZE=parseInt(process.env.GENERATION_BATCH_SIZE||'25',10)
// Target difficulty mix used whenever the admin doesn't explicitly ask for a specific
// difficulty. Must roughly sum to 1 (they don't need to be exact — buildDifficultyPlan
// normalizes via floor+remainder distribution). Override via env if a different split
// (e.g. more Moderate-leaning, closer to a real exam paper) is wanted later.
const DIFFICULTY_MIX_EASY=parseFloat(process.env.DIFFICULTY_MIX_EASY||'0.33')
const DIFFICULTY_MIX_MODERATE=parseFloat(process.env.DIFFICULTY_MIX_MODERATE||'0.34')
const DIFFICULTY_MIX_DIFFICULT=parseFloat(process.env.DIFFICULTY_MIX_DIFFICULT||'0.33')
const TOPIC_CONCURRENCY=parseInt(process.env.TOPIC_CONCURRENCY||'2',10)
const QDRANT_UPSERT_BATCH_SIZE=parseInt(process.env.QDRANT_UPSERT_BATCH_SIZE||'64',10)
const GENERATED_DEDUP_TOP_K=parseInt(process.env.GENERATED_DEDUP_TOP_K||'15',10)
const SAVE_EMBEDDING_BATCH_SIZE=parseInt(process.env.SAVE_EMBEDDING_BATCH_SIZE||'64',10)
const SSE_HEARTBEAT_MS=parseInt(process.env.SSE_HEARTBEAT_MS||'10000',10)

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
topics:[mongoose.Schema.Types.Mixed],
requestedCount:Number,
generatedCount:Number,
partial:{type:Boolean,default:false},
limitedToQuestionLimit:Boolean,
questionLimit:Number,
pyqReferencesUsed:Number,
knowledgeChunksUsed:Number,
questions:[mongoose.Schema.Types.Mixed],
model:String,
savedToQdrant:Number,
stats:mongoose.Schema.Types.Mixed
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

// ---------------------------------------------------------------------------
// Predicted-questions cluster (separate Mongo cluster from chat history).
// Schema matches exactly what the app/frontend expects. Notes on the two
// small deviations from the schema you pasted:
//  - No custom `_id: Number` field: Mongo's default ObjectId is used instead,
//    so nothing needs to be generated/tracked manually on our side.
//  - `exam`, `year` and `paper` are kept (they're useful for filtering by
//    exam category later) but are no longer `required` — generated/predicted
//    questions aren't tied to a specific past sitting the way PYQs are.
// ---------------------------------------------------------------------------
const predictedQuestionSchema=new mongoose.Schema({
exam:{type:String,trim:true},
year:{type:Number},
paper:{type:String,trim:true},
subject:{type:String,required:true,trim:true},
topic:{type:String,trim:true},
imageUrl:{type:String,trim:true,default:null},
english:{
question:{type:String,required:true},
options:{type:Object,required:true},
english_explanation:{type:String,trim:true,default:''}
},
hindi:{
question:{type:String,required:true},
options:{type:Object,required:true},
hindi_explanation:{type:String,trim:true,default:''}
},
marks:{type:Number,default:2},
negativeMarks:{type:Number,default:0.66},
correct_answer:{type:Number,required:true},
batchId:{type:String,trim:true,index:true}
},{timestamps:true})

let predictQuesConnection=null
let predictQuesConnectPromise=null
let lastPredictQuesError=null
let PredictedQuestion=null

function initPredictQuesConnection(){
if(predictQuesConnection) return
predictQuesConnection=mongoose.createConnection(PREDICTQUES_URI,{serverSelectionTimeoutMS:8000})
predictQuesConnection.on('connected',()=>{
console.log('[predictQues] connected')
lastPredictQuesError=null
})
predictQuesConnection.on('error',e=>{
console.error('[predictQues] connection error',e.message)
lastPredictQuesError=e.message
})
predictQuesConnection.on('disconnected',()=>{
console.log('[predictQues] disconnected')
})
PredictedQuestion=predictQuesConnection.model('PredictedQuestion',predictedQuestionSchema,'predicted_questions')
}

function connectPredictQues(){
if(!PREDICTQUES_URI) return Promise.resolve(false)
if(!predictQuesConnection) initPredictQuesConnection()
if(predictQuesConnection.readyState===1) return Promise.resolve(true)
if(!predictQuesConnectPromise){
predictQuesConnectPromise=predictQuesConnection.asPromise()
.then(()=>{
lastPredictQuesError=null
return true
})
.catch(e=>{
console.error('[predictQues connect]',e.message)
lastPredictQuesError=e.message
predictQuesConnectPromise=null
return false
})
}
return predictQuesConnectPromise
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

async function embedTexts(texts,batchSize){
if(!texts||!texts.length) return []
const embedder=await getEmbedder()
const out=[]
for await(const batch of embedder.embed(texts,batchSize||32)){
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

async function searchGeneratedQuestions(vector,topK){
try{
const result=await qdrant.query(QDRANT_GENERATED_QUESTIONS_COLLECTION,{
query:vector,
limit:topK,
with_payload:true
})
return result.points||[]
}catch(e){
return []
}
}

let generatedCollectionEnsured=false

// Checks (and creates if missing) the generated_questions collection.
// IMPORTANT: this now also validates that an *existing* collection's vector
// size actually matches what the current embedding model produces. A silent
// mismatch here (e.g. collection created under a different EMBEDDING_MODEL_NAME
// in the past) is the most common cause of every upsert failing with a cryptic
// Qdrant error, so we surface it as a clear, descriptive error instead.
async function ensureGeneratedQuestionsCollection(vectorSize,log){
const debugLog=log||(()=>{})
if(generatedCollectionEnsured) return
try{
const info=await qdrant.getCollection(QDRANT_GENERATED_QUESTIONS_COLLECTION)
const existingSize=(info&&info.config&&info.config.params&&info.config.params.vectors)?info.config.params.vectors.size:null
debugLog(`Collection "${QDRANT_GENERATED_QUESTIONS_COLLECTION}" already exists (vector size ${existingSize})`)
if(existingSize!=null&&existingSize!==vectorSize){
throw new Error(`Vector size mismatch: collection "${QDRANT_GENERATED_QUESTIONS_COLLECTION}" was created with ${existingSize} dims, but the current embedding model "${EMBEDDING_MODEL_NAME}" produces ${vectorSize} dims. Either recreate the collection or fix EMBEDDING_MODEL_NAME.`)
}
generatedCollectionEnsured=true
return
}catch(e){
if(/Vector size mismatch/.test(e.message||'')) throw e
debugLog(`Collection "${QDRANT_GENERATED_QUESTIONS_COLLECTION}" not found or unreadable (${e.message}), attempting to create it`)
}
try{
await qdrant.createCollection(QDRANT_GENERATED_QUESTIONS_COLLECTION,{
vectors:{size:vectorSize,distance:'Cosine'}
})
generatedCollectionEnsured=true
debugLog(`Created collection "${QDRANT_GENERATED_QUESTIONS_COLLECTION}" with vector size ${vectorSize}`)
}catch(e){
if(!/already exists|409/i.test(e.message||'')) throw e
generatedCollectionEnsured=true
debugLog(`Collection "${QDRANT_GENERATED_QUESTIONS_COLLECTION}" already existed on create race, continuing`)
}
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
const system='You are an intent extraction engine for a multi-exam competitive question generation system. It covers many exam categories, including but not limited to UPSC Civil Services, State PCS (BPSC, UPPSC, MPPSC, RPSC, etc.), SSC exams (CGL, CHSL, MTS, GD), Banking exams (IBPS PO/Clerk, SBI PO/Clerk, RBI), Railways (RRB NTPC/Group D), Defence (NDA, CDS), and government teacher recruitment exams (CTET, state TET). Extract structured parameters from the admin natural language request. The request may cover ONE topic or MULTIPLE distinct topics/subjects/exams/difficulty levels in the same message. Always respond with strict JSON only, no markdown, no prose, no code fences.'
const user=`Admin request: "${query}"

Return ONLY a JSON object in this exact shape:
{"requests":[{"count":null,"examType":null,"subject":null,"topic":null,"chapter":null,"keywords":[],"difficulty":null}]}

Rules:
Create ONE object per distinct topic/subject/difficulty combination the admin asked for. If only one topic is mentioned, return an array with exactly one object.
count is the integer number of questions requested for that specific topic, or null if not mentioned or if only a single combined total was given for multiple topics.
examType is the exam name and stage if mentioned, for example "UPSC Prelims", "BPSC", "State PSC Mains", "SSC CGL Tier 1", "IBPS PO", "CTET Paper 1", "RRB NTPC", or null. If mentioned once for the whole request, repeat it on every object.
topic is the specific topic those questions should be about.
chapter is the book chapter or syllabus section if identifiable, otherwise same as topic or null.
subject is the broader subject area such as Polity, History, Geography, Economy, Science, Environment, Reasoning, Quantitative Aptitude, English Language, Current Affairs or Pedagogy, inferred from the topic if not explicit.
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

// Exams where the classic "Statement I / Statement II ... which of the
// statements given above is/are correct" analytical question format is
// standard (UPSC Prelims/Mains and State PCS). Other exam categories (SSC,
// Banking, Railways, Teaching, Defence, etc.) generally use direct,
// single-statement MCQs, so they're excluded from this pattern.
function isStatementBasedExam(examType){
return /\b(upsc|ias|ips|ifs|civil\s*services?|\bpcs\b|bpsc|uppsc|mppsc|rpsc|opsc|wbpsc|jpsc|hpsc|mpsc|tnpsc|kpsc|state\s*psc|state\s*service\s*commission)\b/i.test(examType||'')
}

// Roughly 4-5 long statement-based questions per batch, only for
// UPSC/PCS-style exams, and only when the batch is big enough for that to
// make sense.
function computeStatementCount(batchCount,statementEligible){
if(!statementEligible) return 0
if(!batchCount||batchCount<5) return 0
return clamp(Math.round(batchCount*0.18),4,5)
}

const DIFFICULTY_LEVELS=['Easy','Moderate','Difficult']

// Builds an exact, shuffled, per-question difficulty assignment for a batch.
// If the admin explicitly asked for a difficulty, every slot is that value
// (unchanged behaviour). Otherwise the batch is split across Easy/Moderate/
// Difficult according to DIFFICULTY_MIX_* and randomly ordered, so results
// aren't skewed toward one level (this used to happen because the prompt's
// fallback text literally said "Moderate, matching the exam standard" when
// no difficulty was given — that's what was causing ~78% Moderate).
function buildDifficultyPlan(count,fixedDifficulty){
if(!count||count<1) return []
const normalizedFixed=(fixedDifficulty||'').trim()
if(normalizedFixed) return new Array(count).fill(normalizedFixed)

const ratios={Easy:DIFFICULTY_MIX_EASY,Moderate:DIFFICULTY_MIX_MODERATE,Difficult:DIFFICULTY_MIX_DIFFICULT}
const raw=DIFFICULTY_LEVELS.map(level=>count*(ratios[level]||0))
const counts=raw.map(n=>Math.floor(n))
let remainder=count-counts.reduce((a,b)=>a+b,0)
const fracOrder=raw
.map((n,i)=>({i,frac:n-Math.floor(n)}))
.sort((a,b)=>b.frac-a.frac)
let cursor=0
while(remainder>0&&fracOrder.length){
counts[fracOrder[cursor%fracOrder.length].i]++
remainder--
cursor++
}

const plan=[]
DIFFICULTY_LEVELS.forEach((level,i)=>{
for(let k=0;k<counts[i];k++) plan.push(level)
})

for(let i=plan.length-1;i>0;i--){
const j=crypto.randomInt(0,i+1)
;[plan[i],plan[j]]=[plan[j],plan[i]]
}
return plan
}

// Fisher-Yates-shuffles a question's option VALUES across the fixed A/B/C/D
// key set (English and Hindi kept in lockstep so both languages still refer
// to the same underlying option), and updates correct_answer to point at
// wherever the correct value landed. This is a hard programmatic guarantee
// against the "correct answer is almost always A" bias models tend to have —
// prompting alone (asking the model to "vary the answer") is not reliable
// enough on its own, so this runs on every accepted question regardless of
// what the model produced.
function shuffleQuestionOptions(q){
const keys=Object.keys(q.options||{})
if(keys.length<2) return q
const correctKey=(q.correct_answer||'').trim().toUpperCase()
const correctIndex=keys.indexOf(correctKey)
if(correctIndex===-1) return q

const order=keys.map((_,i)=>i)
for(let i=order.length-1;i>0;i--){
const j=crypto.randomInt(0,i+1)
;[order[i],order[j]]=[order[j],order[i]]
}

const hasHindi=q.hindi_options&&typeof q.hindi_options==='object'
const newOptions={}
const newHindiOptions=hasHindi?{}:null
keys.forEach((key,i)=>{
const sourceKey=keys[order[i]]
newOptions[key]=q.options[sourceKey]
if(newHindiOptions) newHindiOptions[key]=q.hindi_options[sourceKey]
})

q.options=newOptions
if(newHindiOptions) q.hindi_options=newHindiOptions
const newCorrectPos=order.indexOf(correctIndex)
q.correct_answer=keys[newCorrectPos]
return q
}

function computeQuestionStats(questions){
const answerCounts={A:0,B:0,C:0,D:0}
const difficultyCounts={}
questions.forEach(q=>{
const ans=(q.correct_answer||'').trim().toUpperCase()
if(answerCounts[ans]!==undefined) answerCounts[ans]++
const diff=q.difficulty||'Unspecified'
difficultyCounts[diff]=(difficultyCounts[diff]||0)+1
})
return {answerCounts,difficultyCounts,total:questions.length}
}

function buildPrompt({examType,topic,subject,difficulty,batchCount,pyqText,kbText,avoidList,difficultyPlan}){
const exam=examType||'Competitive Exam'
const statementEligible=isStatementBasedExam(examType)
const statementCount=computeStatementCount(batchCount,statementEligible)

const plan=(difficultyPlan&&difficultyPlan.length===batchCount)?difficultyPlan:buildDifficultyPlan(batchCount,difficulty)
const difficultyList=plan.map((d,i)=>`${i+1}. ${d}`).join('\n')
const difficultyIsFixed=!!(difficulty||'').trim()

const styleBlock=statementEligible&&statementCount>0
?`This is a ${exam} style paper. Out of the ${batchCount} questions, exactly ${statementCount} must be long, analytical statement-based questions in the classic UPSC/State PCS pattern: present 2 to 4 numbered statements (Statement I, Statement II, ...) about the topic, and ask something like "Which of the statements given above is/are correct?" with options such as "A) 1 only  B) 2 only  C) Both 1 and 2  D) Neither 1 nor 2". Set "statement_based":true on those questions only. The remaining ${batchCount-statementCount} questions must be regular, direct, single-statement MCQs with "statement_based":false.`
:`This is a ${exam} style paper. Keep every question a direct, single-statement factual or conceptual MCQ typical of ${exam} — do NOT use the multi-statement "which of the statements given above is/are correct" format. Set "statement_based":false on every question.`

const system=`You are a senior ${exam} question setter writing for an exam-prep platform that serves many exam categories (UPSC, State PCS, SSC, Banking, Railways, Teaching, Defence, and more), so match the tone, length and difficulty conventions of the specific exam named below rather than defaulting to a UPSC style. Write fresh, original MCQs — never copy or lightly reword the sample previous-year questions below; use them only to match style, tone and difficulty. Use the knowledge base text as the sole factual source. Also provide an accurate, natural Hindi translation of the question, every option, and the explanation — real translation, not transliteration; commonly-used technical terms, proper nouns and numerals may stay as they conventionally appear in Hindi exam papers. Write each explanation (English and Hindi) as a direct, self-contained statement of fact — never open with a meta-phrase like "As per the knowledge," "Based on the provided context," or "इस संदर्भ के अनुसार". Vary which option letter (A, B, C or D) holds the correct answer from question to question — never default to A out of habit. Output NDJSON only: exactly one valid JSON object per line, no surrounding array brackets, no commas between lines, no blank lines, no markdown, no code fences, no numbering, no text before or after the lines.`

const avoidBlock=(avoidList&&avoidList.length)?`\nDo not repeat or closely rephrase any of these already-used question stems:\n${avoidList.map(s=>`- ${s}`).join('\n')}\n`:''

const difficultyBlock=difficultyIsFixed
?`Difficulty: every question must be "${difficulty}".`
:`Required difficulty for each question, in order (line 1 = first question, line 2 = second question, etc.) — this was requested as a balanced, unspecified mix, so match it exactly rather than defaulting to Moderate:
${difficultyList}
An "Easy" question should be answerable from direct recall of a single fact. A "Moderate" question should require connecting two related facts or a short inference. A "Difficult" question should require multi-step reasoning, fine distinctions between close options, or synthesis across sub-topics — not just longer wording.`

const user=`Topic: ${topic}
Subject: ${subject||'General Studies'}
Exam: ${exam}
${difficultyBlock}
${styleBlock}
Generate exactly ${batchCount} new original MCQs, in the same order as the difficulty list above. Output exactly ${batchCount} lines, each a standalone JSON object in this exact shape:
{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"correct_answer":"A","explanation":"...","hindi_question":"...","hindi_options":{"A":"...","B":"...","C":"...","D":"..."},"hindi_explanation":"...","statement_based":false,"difficulty":"Easy|Moderate|Difficult","topic":"${topic}","subject":"${subject||''}"}

Rules:
- "options" and "hindi_options" must have exactly the same four keys (A, B, C, D) in the same order; each hindi_options value is the Hindi translation of the matching options value.
- "correct_answer" is always the English option letter (A, B, C or D). Spread correct answers roughly evenly across A, B, C and D over the ${batchCount} questions — do not cluster them on one letter.
- The "difficulty" field on each line must match its required difficulty above.
- "hindi_question" and "hindi_explanation" must never be empty — always fill them in.
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
if(!obj||typeof obj.question!=='string'||!obj.question.trim()) return null
if(!obj.options||typeof obj.options!=='object'||!Object.keys(obj.options).length) return null
// Bilingual is now mandatory. If the model skipped the Hindi fields on a
// given line, reject the line — the existing retry loop in
// generateQuestionsForBatch will simply ask for that many more questions,
// so nothing is lost, it just costs one more attempt.
if(typeof obj.hindi_question!=='string'||!obj.hindi_question.trim()) return null
if(!obj.hindi_options||typeof obj.hindi_options!=='object'||!Object.keys(obj.hindi_options).length) return null
shuffleQuestionOptions(obj)
return obj
}catch(e){
return null
}
}

function normalizeStem(text){
return (text||'').toLowerCase().replace(/[^a-z0-9\s]/g,'').replace(/\s+/g,' ').trim().slice(0,120)
}

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

async function generateQuestionsForBatch(params,dedupState,onQuestion){
let stillNeeded=params.batchCount
let collected=[]
// Built once, sized to the full batch target, and consumed one slot per
// ACCEPTED question (not per raw line) — so it stays exactly in sync with
// `collected` across retries regardless of how many lines get rejected or
// deduped along the way. This is what guarantees the final saved batch has
// a balanced difficulty spread instead of drifting toward whatever the
// model defaults to.
const difficultyPlan=buildDifficultyPlan(params.batchCount,params.difficulty)
let planCursor=0
for(let attempt=0;attempt<3&&stillNeeded>0;attempt++){
const avoidList=[
...dedupState.historicalTexts.slice(-10),
...dedupState.recentTexts.slice(-60)
]
const passParams={...params,batchCount:stillNeeded,avoidList,difficultyPlan:difficultyPlan.slice(planCursor)}
// Bilingual output (English + Hindi) roughly doubles the tokens per
// question versus English-only, so the per-question budget is raised
// accordingly (was *230+300).
const maxTokens=clamp(stillNeeded*450+400,800,24000)
const {system,user}=buildPrompt(passParams)
const acc=createLineAccumulator(line=>{
const q=parseQuestionLine(line)
if(!q) return
const stem=normalizeStem(q.question)
if(dedupState.seen.has(stem)) return
dedupState.seen.add(stem)
dedupState.recentTexts.push(q.question.slice(0,140))
q.difficulty=difficultyPlan[planCursor]||q.difficulty||'Moderate'
planCursor++
collected.push(q)
if(onQuestion) onQuestion(q)
})
try{
await streamPSModel(system,user,delta=>acc.push(delta),maxTokens)
}catch(e){
}
acc.flush()
stillNeeded=params.batchCount-collected.length
}
return collected
}

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

async function generateForTopic(spec,sendEvent,topicIndex,totalTopics,dedupState){
let pyqPoints,kbPoints
try{
const searchText=buildSearchText(spec)
const queryVector=await embedOne(searchText)
const [pyq,kb,pastGenerated]=await Promise.all([
searchQuestionBank(queryVector,QUESTION_BANK_TOP_K),
searchKnowledgeBase(queryVector,KNOWLEDGE_BASE_TOP_K),
searchGeneratedQuestions(queryVector,GENERATED_DEDUP_TOP_K)
])
pyqPoints=pyq
kbPoints=kb
for(const p of pastGenerated){
const stem=normalizeStem(p.payload&&p.payload.question)
if(!stem||dedupState.seen.has(stem)) continue
dedupState.seen.add(stem)
dedupState.historicalTexts.push((p.payload.question||'').slice(0,140))
}
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
batchQuestions=[]
}

sendEvent('batch_done',{topicIndex,totalTopics,topic:spec.topic,batch:b+1,totalBatches:batches.length,delivered:batchQuestions.length,requested:batchCount})

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

// Saves generated questions to Qdrant with full debug instrumentation.
// Returns {savedCount, totalQuestions, debug} — debug always contains a
// per-batch/per-chunk breakdown plus any errors encountered, so it can be
// surfaced directly in an API response or an SSE event and inspected in
// the browser Network tab instead of only appearing in server logs.
// NOTE: this is now ONLY ever called manually via
// POST /api/questions/save-to-qdrant — the /generate flow no longer calls
// this automatically.
async function saveGeneratedQuestions(questions,meta){
const debug={
requestId:meta.requestId||null,
collection:QDRANT_GENERATED_QUESTIONS_COLLECTION,
embeddingModel:EMBEDDING_MODEL_NAME,
totalQuestions:questions.length,
startedAt:new Date().toISOString(),
finishedAt:null,
savedCount:0,
skipped:null,
fatalError:null,
batches:[],
errors:[],
logs:[]
}
const log=(msg)=>{
console.log(`[saveGeneratedQuestions]${meta.requestId?` [${meta.requestId}]`:''} ${msg}`)
debug.logs.push(msg)
}

if(!questions.length){
debug.skipped='No questions were passed to save'
debug.finishedAt=new Date().toISOString()
log(debug.skipped)
return {savedCount:0,totalQuestions:0,debug}
}

log(`Starting save of ${questions.length} questions to "${QDRANT_GENERATED_QUESTIONS_COLLECTION}"`)

let embedder
try{
embedder=await getEmbedder()
}catch(e){
log(`Failed to initialize embedder: ${e.message}`)
debug.fatalError=`embedder_init: ${e.message}`
debug.finishedAt=new Date().toISOString()
const err=new Error(`Failed to initialize embedder: ${e.message}`)
err.debug=debug
throw err
}

const texts=questions.map(q=>[q.question,...Object.values(q.options||{})].join(' '))
const upsertPromises=[]
let offset=0
let batchIndex=0

try{
for await(const vectorBatch of embedder.embed(texts,SAVE_EMBEDDING_BATCH_SIZE)){
batchIndex++
const batchStartedAt=Date.now()
const batchQuestions=questions.slice(offset,offset+vectorBatch.length)
offset+=vectorBatch.length
const vectorSize=vectorBatch.length?vectorBatch[0].length:null
log(`Batch ${batchIndex}: embedded ${batchQuestions.length} questions (dim=${vectorSize})`)

try{
if(!generatedCollectionEnsured) await ensureGeneratedQuestionsCollection(vectorSize,log)
}catch(e){
log(`Batch ${batchIndex}: collection check/create failed: ${e.message}`)
debug.errors.push({stage:'ensure_collection',batch:batchIndex,error:e.message})
throw e
}

const points=batchQuestions.map((q,j)=>({
id:crypto.randomUUID(),
vector:Array.from(vectorBatch[j]),
payload:{
exam:q.examType||meta.examType||null,
subject:q.subject||meta.subject||null,
topic:q.topic||meta.topic||null,
chapter:q.chapter||meta.chapter||null,
question:q.question,
options:q.options,
correct_answer:q.correct_answer||null,
explanation:q.explanation||null,
hindi_question:q.hindi_question||null,
hindi_options:q.hindi_options||null,
hindi_explanation:q.hindi_explanation||null,
statement_based:!!q.statement_based,
difficulty:q.difficulty||meta.difficulty||null,
source:'generated',
generated_at:new Date().toISOString(),
request_id:meta.requestId||null
}
}))

const batchDebug={batchIndex,questionCount:batchQuestions.length,vectorSize,chunks:[]}
for(let i=0;i<points.length;i+=QDRANT_UPSERT_BATCH_SIZE){
const chunk=points.slice(i,i+QDRANT_UPSERT_BATCH_SIZE)
const chunkIndex=Math.floor(i/QDRANT_UPSERT_BATCH_SIZE)
upsertPromises.push(
qdrant.upsert(QDRANT_GENERATED_QUESTIONS_COLLECTION,{wait:true,points:chunk})
.then(()=>{
log(`Batch ${batchIndex} chunk ${chunkIndex}: upserted ${chunk.length} points`)
batchDebug.chunks.push({chunkIndex,requested:chunk.length,saved:chunk.length,error:null})
return chunk.length
})
.catch(e=>{
const errMsg=e.message||'Unknown upsert error'
log(`Batch ${batchIndex} chunk ${chunkIndex}: FAILED - ${errMsg}`)
debug.errors.push({stage:'upsert',batch:batchIndex,chunk:chunkIndex,error:errMsg})
batchDebug.chunks.push({chunkIndex,requested:chunk.length,saved:0,error:errMsg})
return 0
})
)
}
batchDebug.ms=Date.now()-batchStartedAt
debug.batches.push(batchDebug)
}
}catch(e){
const counts=await Promise.all(upsertPromises)
debug.savedCount=counts.reduce((a,b)=>a+b,0)
debug.fatalError=e.message
debug.finishedAt=new Date().toISOString()
log(`Fatal error, stopping after saving ${debug.savedCount}/${questions.length}: ${e.message}`)
const err=new Error(e.message)
err.debug=debug
throw err
}

const counts=await Promise.all(upsertPromises)
const savedCount=counts.reduce((a,b)=>a+b,0)
debug.savedCount=savedCount
debug.finishedAt=new Date().toISOString()
log(`Finished: saved ${savedCount}/${questions.length}`)
return {savedCount,totalQuestions:questions.length,debug}
}

async function deleteGeneratedQuestionsByRequestId(requestId){
if(!requestId) return 0
const result=await qdrant.delete(QDRANT_GENERATED_QUESTIONS_COLLECTION,{
filter:{must:[{key:'request_id',match:{value:requestId}}]},
wait:true
})
return result
}

// Maps the model's English option letter (A/B/C/D) onto a numeric index for
// the predicted-questions schema, where correct_answer is a Number. Keys are
// sorted first so the mapping is stable even if key insertion order ever
// varies. See PREDICTED_ANSWER_INDEX_BASE above to switch 0-based/1-based.
function letterToOptionIndex(letter,optionsObj){
if(!letter||!optionsObj) return null
const keys=Object.keys(optionsObj).sort()
const pos=keys.indexOf(String(letter).trim().toUpperCase())
if(pos===-1) return null
return pos+PREDICTED_ANSWER_INDEX_BASE
}

// Transforms one generated question (our internal NDJSON shape) into a
// document matching the predicted-questions Mongo schema.
function buildPredictedDoc(q,meta,batchId){
const englishOptions=q.options||{}
const hindiOptions=q.hindi_options||{}
return {
exam:q.examType||meta.examType||null,
subject:q.subject||meta.subject||null,
topic:q.topic||meta.topic||null,
imageUrl:q.imageUrl||null,
english:{
question:q.question||'',
options:englishOptions,
english_explanation:q.explanation||''
},
hindi:{
question:q.hindi_question||'',
options:hindiOptions,
hindi_explanation:q.hindi_explanation||''
},
marks:Number.isFinite(q.marks)?q.marks:(Number.isFinite(meta.marks)?meta.marks:2),
negativeMarks:Number.isFinite(q.negativeMarks)?q.negativeMarks:(Number.isFinite(meta.negativeMarks)?meta.negativeMarks:0.66),
correct_answer:letterToOptionIndex(q.correct_answer,englishOptions),
batchId:q.batchId||batchId||meta.batchId||null
}
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
const predictQuesConnected=await connectPredictQues()
res.json({
ok:true,
time:new Date().toISOString(),
collections:collections.collections.map(c=>c.name),
mongoConfigured:!!PSMODELCHATHISDB_URI,
mongo:mongoConnected?'connected':'disconnected',
mongoError:lastMongoError,
predictQuesConfigured:!!PREDICTQUES_URI,
predictQues:predictQuesConnected?'connected':'disconnected',
predictQuesError:lastPredictQuesError,
questionLimit:question_limit,
maxTopics:MAX_TOPICS,
generationBatchSize:GENERATION_BATCH_SIZE,
topicConcurrency:TOPIC_CONCURRENCY
})
}catch(e){
res.status(500).json({ok:false,error:e.message})
}
})

// Debug helper: tells you whether the generated_questions collection exists,
// what vector size it was created with, and what vector size the current
// embedding model actually produces. If sizesMatch is false, that's why
// saves are failing — recreate the collection or fix EMBEDDING_MODEL_NAME.
app.get('/api/questions/collection-info',requireAdmin,async(req,res)=>{
try{
let exists=true
let info=null
try{
info=await qdrant.getCollection(QDRANT_GENERATED_QUESTIONS_COLLECTION)
}catch(e){
exists=false
}
let embeddingDim=null
let embedProbeError=null
try{
const vectors=await embedTexts(['__dimension_probe__'])
embeddingDim=vectors[0]?vectors[0].length:null
}catch(e){
embedProbeError=e.message
}
const vectorSizeInCollection=(info&&info.config&&info.config.params&&info.config.params.vectors)?info.config.params.vectors.size:null
res.json({
collection:QDRANT_GENERATED_QUESTIONS_COLLECTION,
exists,
pointsCount:(info&&typeof info.points_count!=='undefined')?info.points_count:null,
vectorSizeInCollection,
embeddingModel:EMBEDDING_MODEL_NAME,
embeddingDim,
embedProbeError,
sizesMatch:(exists&&vectorSizeInCollection!=null&&embeddingDim!=null)?vectorSizeInCollection===embeddingDim:null,
upsertBatchSize:QDRANT_UPSERT_BATCH_SIZE,
saveEmbeddingBatchSize:SAVE_EMBEDDING_BATCH_SIZE
})
}catch(e){
res.status(500).json({error:e.message||'Internal error'})
}
})

// Dedicated endpoint to (re)save generated questions to Qdrant, independent
// of the /generate SSE flow. This is now the ONLY way questions land in the
// generated_questions Qdrant collection — /generate no longer does this
// automatically. Pass either:
//  - { requestId } to load questions from Mongo chat history and save those, or
//  - { questions:[...], requestId?, examType?, subject?, topic?, chapter?, difficulty? }
// The full debug breakdown (per-batch, per-chunk, errors) is returned in the
// JSON response so it is visible in the Network tab.
app.post('/api/questions/save-to-qdrant',requireAdmin,async(req,res)=>{
const body=req.body||{}
const requestId=(body.requestId||'').trim()||null
let questions=Array.isArray(body.questions)?body.questions:null
let meta={
requestId,
examType:body.examType||null,
subject:body.subject||null,
topic:body.topic||null,
chapter:body.chapter||null,
difficulty:body.difficulty||null
}
let source='body'

try{
if(!questions&&requestId){
source='mongo'
if(!(await connectMongo())){
return res.status(503).json({error:'MongoDB not configured or unavailable, cannot look up questions by requestId. Pass "questions" directly instead.'})
}
const doc=await ChatHistory.findOne({requestId}).lean()
if(!doc) return res.status(404).json({error:`No chat history found for requestId "${requestId}"`})
questions=doc.questions||[]
meta={
requestId,
examType:doc.examType||null,
subject:doc.subject||null,
topic:doc.topic||null,
chapter:doc.chapter||null,
difficulty:doc.difficulty||null
}
}

if(!questions||!questions.length){
return res.status(400).json({error:'No questions to save. Provide a non-empty "questions" array, or a "requestId" that has stored questions in chat history.',source})
}

const result=await saveGeneratedQuestions(questions,meta)

let mongoUpdateError=null
if(requestId){
try{
if(await connectMongo()){
await ChatHistory.updateOne({requestId},{$set:{savedToQdrant:result.savedCount}})
}
}catch(e){
mongoUpdateError=e.message
console.error('[save-to-qdrant mongo backfill]',e.message)
}
}

res.json({
requestId,
source,
totalQuestions:result.totalQuestions,
savedCount:result.savedCount,
fullyPersisted:result.savedCount===result.totalQuestions,
mongoUpdateError,
debug:result.debug
})
}catch(e){
console.error('[save-to-qdrant]',e)
res.status(500).json({
error:e.message||'Internal error while saving to Qdrant',
source,
debug:e.debug||null
})
}
})

// Admin-triggered upload of reviewed/generated questions into the predicted-
// questions Mongo cluster (PREDICTQUES_URI). This is a deliberate, manual
// action from the admin panel — nothing gets pushed here automatically.
// Pass either:
//  - { requestId, batchId?, examType?, subject?, topic?, marks?, negativeMarks? }
//    to load questions from chat history and upload those, or
//  - { questions:[...], batchId?, examType?, subject?, topic?, marks?, negativeMarks? }
//    to upload an admin-edited/curated list directly.
// Each question is validated against the predicted-questions schema
// (English + Hindi content, resolvable correct_answer, subject) before
// insertion; anything that fails validation is skipped and reported back
// under "skipped" rather than silently dropped.
app.post('/api/questions/upload-to-predicted',requireAdmin,async(req,res)=>{
const body=req.body||{}
const requestId=(body.requestId||'').trim()||null
let questions=Array.isArray(body.questions)?body.questions:null
let meta={
examType:body.examType||null,
subject:body.subject||null,
topic:body.topic||null,
marks:Number.isFinite(parseFloat(body.marks))?parseFloat(body.marks):null,
negativeMarks:Number.isFinite(parseFloat(body.negativeMarks))?parseFloat(body.negativeMarks):null
}
const batchId=(body.batchId||'').trim()||requestId||crypto.randomUUID()
let source='body'

try{
if(!PREDICTQUES_URI){
return res.status(500).json({error:'PREDICTQUES_URI is not configured on the server. Add it as an env var (the predicted-questions cluster connection string) and redeploy.'})
}

if(!questions&&requestId){
source='mongo'
if(!(await connectMongo())){
return res.status(503).json({error:'MongoDB (chat history) not configured or unavailable, cannot look up questions by requestId. Pass "questions" directly instead.'})
}
const doc=await ChatHistory.findOne({requestId}).lean()
if(!doc) return res.status(404).json({error:`No chat history found for requestId "${requestId}"`})
questions=doc.questions||[]
meta={
examType:meta.examType||doc.examType||null,
subject:meta.subject||doc.subject||null,
topic:meta.topic||doc.topic||null,
marks:meta.marks,
negativeMarks:meta.negativeMarks
}
}

if(!questions||!questions.length){
return res.status(400).json({error:'No questions to upload. Provide a non-empty "questions" array, or a "requestId" that has stored questions in chat history.',source})
}

if(!(await connectPredictQues())){
return res.status(503).json({error:`Could not connect to the predicted-questions cluster: ${lastPredictQuesError||'unknown connection error'}`})
}

const docs=[]
const skipped=[]
questions.forEach((q,i)=>{
const doc=buildPredictedDoc(q,meta,batchId)
const reasons=[]
if(!doc.subject) reasons.push('missing subject')
if(!doc.english.question) reasons.push('missing English question')
if(!doc.english.options||!Object.keys(doc.english.options).length) reasons.push('missing English options')
if(!doc.hindi.question) reasons.push('missing Hindi question (not translated — regenerate or add translation before uploading)')
if(!doc.hindi.options||!Object.keys(doc.hindi.options).length) reasons.push('missing Hindi options')
if(doc.correct_answer===null||doc.correct_answer===undefined) reasons.push('could not resolve correct_answer to an option index')
if(reasons.length){
skipped.push({index:i,question:(q.question||'').slice(0,80),reasons})
return
}
docs.push(doc)
})

if(!docs.length){
return res.status(400).json({error:'None of the supplied questions passed validation for the predicted-questions schema.',source,batchId,skipped})
}

let insertedCount=docs.length
let insertErrors=[]
try{
await PredictedQuestion.insertMany(docs,{ordered:false})
}catch(e){
if(e&&Array.isArray(e.writeErrors)&&e.writeErrors.length){
insertErrors=e.writeErrors.map(we=>({index:we.index,error:(we.errmsg)||(we.err&&we.err.errmsg)||'insert failed'}))
insertedCount=docs.length-insertErrors.length
}else if(e&&e.insertedDocs){
insertedCount=e.insertedDocs.length
insertErrors=[{error:e.message}]
}else{
throw e
}
}

res.json({
requestId,
source,
batchId,
totalReceived:questions.length,
validated:docs.length,
uploaded:insertedCount,
skippedValidationCount:skipped.length,
skipped,
insertErrors
})
}catch(e){
console.error('[upload-to-predicted]',e)
res.status(500).json({error:e.message||'Internal error while uploading to the predicted-questions cluster',source,batchId})
}
})

app.post('/api/questions/generate',requireAdmin,async(req,res)=>{
const body=req.body||{}
const query=(body.query||'').trim()

const explicitTopics=Array.isArray(body.topics)?body.topics.map(normalizeSpec).filter(s=>s.topic):null

if(!query&&!(body.topic||'').trim()&&!(explicitTopics&&explicitTopics.length)){
return res.status(400).json({error:'query, topic, or topics is required'})
}

res.setHeader('Content-Type','text/event-stream')
res.setHeader('Cache-Control','no-cache')
res.setHeader('Connection','keep-alive')
res.setHeader('X-Accel-Buffering','no')
res.flushHeaders()

res.on('error',(err)=>{
console.error('[response stream error]',err.message)
})

function sendEvent(event,data){
try{
res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
}catch(e){
console.error('[sendEvent write failed]',e.message)
}
}

const heartbeat=setInterval(()=>{
try{
res.write(': keep-alive\n\n')
}catch(e){
}
},SSE_HEARTBEAT_MS)

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
batchId:requestId,
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

const dedupState={seen:new Set(),recentTexts:[],historicalTexts:[]}
const results=await runWithConcurrency(
specs,
TOPIC_CONCURRENCY,
(spec,i)=>generateForTopic(spec,sendEvent,i+1,specs.length,dedupState)
)

let questions=results.flatMap(r=>r.questions||[])
const pyqReferencesUsed=results.reduce((a,r)=>a+(r.pyqReferencesUsed||0),0)
const knowledgeChunksUsed=results.reduce((a,r)=>a+(r.knowledgeChunksUsed||0),0)

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

const stats=computeQuestionStats(questions)

sendEvent('done',{
requestId,
batchId:requestId,
generatedCount:questions.length,
totalTopics:specs.length,
partial,
topics:topicsResult,
questions,
stats
})

// Chat history is still saved automatically for audit/history purposes.
// Qdrant (generated_questions) and the predicted-questions cluster are
// NOT touched automatically anymore — use POST /api/questions/save-to-qdrant
// and POST /api/questions/upload-to-predicted respectively once the admin
// has reviewed the batch.
let mongoId=null
let mongoError=null
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
savedToQdrant:0,
stats
})
mongoId=doc._id.toString()
}else{
mongoError=PSMODELCHATHISDB_URI
?`MongoDB is configured but not reachable: ${lastMongoError||'connection failed'}`
:'PSMODELCHATHISDB_URI is not set, so chat history cannot be saved.'
}
}catch(e){
mongoError=e.message||'Unknown error saving chat history'
console.error('[mongo save]',mongoError)
}

sendEvent('persisted',{
requestId,
batchId:requestId,
mongoId,
mongoError,
note:'Automatic Qdrant/predicted-questions persistence is disabled. Use POST /api/questions/save-to-qdrant and/or POST /api/questions/upload-to-predicted (with this requestId as batchId) once reviewed.'
})

res.end()
}catch(e){
console.error('[generate]',e)
try{
sendEvent('error',{error:e.message||'Internal error'})
}catch(_){}
res.end()
}finally{
clearInterval(heartbeat)
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
connectPredictQues().then(ok=>{
console.log(ok?'[predictQues] initial connection succeeded':`[predictQues] initial connection failed: ${lastPredictQuesError||'PREDICTQUES_URI not set'}`)
})
})

module.exports=app
