'use strict';
const $ = id => document.getElementById(id);
let token = sessionStorage.getItem('chat-local-token') || '';
let state = null, conversationId = null;
function showError(message = '') { $('error').textContent = message; }
async function api(path, body) {
  const headers = {'Content-Type': 'application/json'};
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', headers,
    body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store'});
  let result;
  try { result = await response.json(); } catch { throw new Error('服务未返回有效 JSON，请检查本机日志。'); }
  if (!response.ok) {
    if (response.status === 401) { token = ''; sessionStorage.removeItem('chat-local-token'); showLogin(); }
    throw new Error(typeof result.detail === 'string' ? result.detail : `请求失败 (${response.status})`);
  }
  return result;
}
function showLogin() { $('login-card').classList.remove('hidden'); $('workspace').classList.add('hidden'); $('logout').classList.add('hidden'); }
function era() { return $('cutoff').value || ($('era').value === 'latest' ? null : $('era').value); }
function bubble(kind, text) { const element = document.createElement('div'); element.className = `bubble ${kind}`; element.textContent = text; $('messages').append(element); $('messages').scrollTop = $('messages').scrollHeight; }
function clearConversation() { conversationId = null; $('messages').replaceChildren(); $('sources').replaceChildren(); $('evidence-status').textContent = '新会话。历史时期筛选与本页会话独立保存。'; }
function setSources(sources, supported) {
  $('sources').replaceChildren();
  $('evidence-status').textContent = sources.length ? `${sources.length} 条真实原话来源；${supported ? '有证据支持对应时期' : '关系阶段尚未确定，不假定亲密或疏远'}。` : '没有找到可核对的相关真实原话；不能据此编造共同经历。';
  for (const source of sources) {
    const details = document.createElement('details'); details.className = 'source';
    const summary = document.createElement('summary'); summary.textContent = `${source.timestamp || source.time || ''} · ${source.raw_sender || source.sender} · 第${source.source_line}行`;
    const quote = document.createElement('pre'); quote.textContent = source.raw_quote || source.quote || source.text || '';
    const meta = document.createElement('p'); meta.className = 'metadata'; meta.textContent = `来源：${source.source_file || '真实聊天记录'}\n类别：${source.category || '真实历史原话'}${source.statement ? '\n提取摘要（原话出处已核验，不等于心理推断已证明）：' + source.statement : ''}`;
    details.append(summary, quote, meta); $('sources').append(details);
  }
}
async function loadConversations() {
  const data = await api(`/chat/api/${state.slug}/conversations`);
  const select = $('conversations'); select.replaceChildren(new Option('选择以继续会话', ''));
  for (const item of data.conversations) select.add(new Option(`${item.updated_at.slice(0,16)} · ${item.era} · ${item.turns}轮`, item.id));
}
async function loadState() {
  state = await api('/chat/api/me');
  $('login-card').classList.add('hidden'); $('workspace').classList.remove('hidden'); $('logout').classList.remove('hidden');
  $('heading').textContent = `${state.name} · 人格分身`;
  $('status').textContent = `真实历史 ${state.history_vectors.toLocaleString()} 个向量；独立会话 ${state.session_memory.turns} 轮 / ${state.session_memory.vectors} 个独立向量。${state.persona_distilled ? '全量画像已蒸馏并整合。' : '人格蒸馏尚未完成，当前候选包不会冒充已完成画像。'}`;
  $('model-status').textContent = state.model.chat_authorized ? `聊天已授权服务：${state.model.service_name} / ${state.model.model}。${state.embedding}` : `模型服务尚未获得日常聊天授权，回复生成会被明确阻止。${state.embedding}`;
  const select = $('era'); select.replaceChildren(new Option('不限时期：截至今天的真实历史', 'latest'));
  const [startYear, startMonth] = (state.history.first_message_at || '2025-10-01').slice(0,7).split('-').map(Number);
  const [endYear, endMonth] = (state.history.last_message_at || '2026-05-01').slice(0,7).split('-').map(Number);
  const start = new Date(startYear,startMonth-1,1), finish = new Date(endYear,endMonth-1,1);
  for (const d = new Date(start); d <= finish; d.setMonth(d.getMonth()+1)) { const month = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`; select.add(new Option(`${month} · 日期筛选，不预设关系`, month)); }
  for (const item of state.eras) select.add(new Option(`${item.label} (${item.start} 至 ${item.end})`, item.id));
  await loadConversations();
}
$('login-form').addEventListener('submit', async event => {
  event.preventDefault(); showError();
  try { const result = await api('/api/auth/login', {username:$('username').value, password:$('password').value}); token=result.token || result.access_token; if (!token) throw new Error('登录接口未返回有效 token'); sessionStorage.setItem('chat-local-token',token); $('password').value=''; await loadState(); }
  catch(error) { showError(error.message); }
});
$('logout').addEventListener('click', async () => { try { await api('/api/auth/logout', {}); } catch {} token=''; sessionStorage.removeItem('chat-local-token'); state=null; clearConversation(); showLogin(); });
$('era').addEventListener('change', () => { $('cutoff').value=''; clearConversation(); });
$('cutoff').addEventListener('change', clearConversation);
$('new-chat').addEventListener('click', clearConversation);
function busy(value) { $('search').disabled=value; $('send').disabled=value; $('new-chat').disabled=value; $('era').disabled=value; $('cutoff').disabled=value; $('conversations').disabled=value; }
async function send(retrievalOnly) {
  const message = $('message').value; if (!message.trim() || !state) return; showError(); busy(true);
  try {
    if (retrievalOnly) {
      const result = await api(`/chat/api/${state.slug}/search`, {query:message,era:era()});
      bubble('retrieval', `【仅检索真实历史 · 未调用LLM】${result.sources.length ? '已在右侧列出可核对的原始来源。' : '没有找到相关证据。'}`); setSources(result.sources,result.supported_relationship_era);
    } else {
      const result = await api(`/chat/api/${state.slug}/chat`, {message,era:era(),conversation_id:conversationId});
      conversationId=result.conversation_id; bubble('user',message); bubble('assistant',result.reply); setSources(result.sources,result.supported_relationship_era); $('message').value='';
      if (!result.memory.indexed) showError('本页对话已保存，但独立对话向量尚未建立；不会影响真实聊天历史，可稍后重建对话索引。');
      await loadConversations();
    }
  } catch(error) { showError(error.message); }
  finally { busy(false); }
}
$('search').addEventListener('click', () => send(true));
$('message-form').addEventListener('submit', event => { event.preventDefault(); send(false); });
$('conversations').addEventListener('change', async () => {
  if (!$('conversations').value) return; showError();
  try {
    const list = await api(`/chat/api/${state.slug}/conversations`), item = list.conversations.find(row => row.id === $('conversations').value);
    if (!item) return; $('cutoff').value=''; if ([...$('era').options].some(option => option.value===item.era)) $('era').value=item.era; else { $('era').value='latest'; $('cutoff').value=item.era; }
    const data = await api(`/chat/api/${state.slug}/conversations/${encodeURIComponent(item.id)}?era=${encodeURIComponent(item.era)}`);
    conversationId=item.id; $('messages').replaceChildren(); for (const message of data.history) bubble(message.role==='user'?'user':'assistant',message.content);
  } catch(error) { showError(error.message); }
});
if (token) loadState().catch(error => { showError(error.message); showLogin(); });
