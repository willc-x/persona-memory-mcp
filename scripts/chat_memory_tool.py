#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""聊天记录 TXT -> ex-memory 结构化输入 + 证据约束人物画像注入。

仅使用 Python 标准库。聊天记录始终在本机处理；不会发送网络请求。

用法：
  python3 chat_memory_tool.py prepare --input chat.txt --target 目标人物 \
      --out ./prepared --expect-lines N
  python3 chat_memory_tool.py inject --ex-skill ./ex-memory/exes/persona/SKILL.md \
      --suhui-dir ./suhui-work/characters/persona
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import os
import json
from pathlib import Path
import re
import shutil
import sys

TS_RE = re.compile(r'^\[(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\][ \t]+(?P<rest>.*)$')
HEADER_TAG_RE = re.compile(
    r'^\[(?P<tag>系统消息|图片|照片|表情|链接/应用|链接|语音|视频|文件|位置|小程序|名片|音乐|红包|转账|网络电话|通话|类型\d+)\][ \t]+(?P<rest>.*)$'
)
SENDER_RE = re.compile(r'^(?P<sender>[^:：]+?)[ \t]*[:：](?P<content>.*)$')
GENERIC_HEADER_RE = re.compile(r'^\[(?P<tag>[^\]\r\n]{1,40})\][ \t]+(?P<rest>.*)$')
PLACEHOLDER_MARKERS = {'图片', '照片', '表情', '链接', '链接/应用', '语音', '视频', '文件', '位置', '小程序', '名片', '音乐', '红包', '转账', '网络电话', '通话'}
PLACEHOLDER_ONLY = re.compile(r'^(?:\s*\[(?:' + '|'.join(re.escape(x) for x in sorted(PLACEHOLDER_MARKERS, key=len, reverse=True)) + r')\]\s*)+$')
BRACKET_GROUP = re.compile(r'^(?:\[[^\]\n]{1,24}\]\s*)+$')
BRACKET_TOKEN = re.compile(r'\[([^\]\n]{1,24})\]')
START_MARK = '<!-- SUHUI_INTEGRATION_START -->'
END_MARK = '<!-- SUHUI_INTEGRATION_END -->'
EXCLUSION_TAGS = {'系统消息'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path.chmod(0o600)


def read_lines(input_file, encoding):
    data = input_file.read_bytes()
    if encoding == 'auto':
        utf_encoding = 'utf-8-sig' if data.startswith(b'\xef\xbb\xbf') else 'utf-8'
        for enc in (utf_encoding, 'gb18030'):
            try:
                return data.decode(enc).splitlines(), enc
            except UnicodeError:
                pass
        raise ValueError('编码无法识别，请指定 --encoding utf-8-sig 或 gb18030')
    used = 'utf-8' if encoding == 'utf-8-sig' and not data.startswith(b'\xef\xbb\xbf') else encoding
    return data.decode(used).splitlines(), used


def content_kind(tag, body):
    """返回类别、溯洄 type、溯洄 kind、是否值得进入 ex-memory 向量库。"""
    if tag == '系统消息':
        return 'system', 'system', 'placeholder', False
    if PLACEHOLDER_ONLY.fullmatch(body) or (not body and tag in PLACEHOLDER_MARKERS):
        label = tag or (BRACKET_TOKEN.findall(body) or [''])[0]
        mtype = {'图片': 'image', '照片': 'image', '语音': 'voice', '视频': 'video'}.get(label, 'file')
        return 'media_only', mtype, 'placeholder', False
    if BRACKET_GROUP.fullmatch(body):
        bracket_terms = BRACKET_TOKEN.findall(body)
        if bracket_terms and all(t not in PLACEHOLDER_MARKERS and t not in EXCLUSION_TAGS for t in bracket_terms):
            return 'named_emoji', 'text', 'emoji', True
    if tag in ('链接/应用', '链接'):
        return 'link_with_text', 'text', 'text', bool(body.strip())
    if tag in PLACEHOLDER_MARKERS:
        return 'media_with_text', 'text', 'text', bool(body.strip())
    return 'text', 'text', 'text', bool(body.strip())


def parse_message_line(raw, known_senders=None):
    """只解析显式的时间/发送者；只去掉冒号后的一个格式分隔空格。"""
    m = TS_RE.fullmatch(raw.lstrip('\ufeff'))
    if not m:
        raise ValueError('缺少有效时间戳或整行格式不匹配')
    date, rest = m.group('date', 'rest')
    try:
        datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    except ValueError as exc:
        raise ValueError('时间戳日期非法') from exc
    tag = ''
    h = HEADER_TAG_RE.match(rest)
    if not h and known_senders:
        candidate = GENERIC_HEADER_RE.match(rest)
        if candidate:
            candidate_sender = SENDER_RE.match(candidate.group('rest'))
            if candidate_sender and candidate_sender.group('sender').strip() in known_senders:
                h = candidate
    if h:
        tag, rest = h.group('tag', 'rest')
    s = SENDER_RE.match(rest)
    if not s:
        raise ValueError('发送者与冒号无法识别')
    after_colon = s.group('content')
    content = after_colon[1:] if after_colon.startswith(' ') else after_colon
    return {'timestamp': date, 'sender': s.group('sender').strip(),
            'raw_sender': s.group('sender'), 'content': content,
            'raw_content_after_colon': after_colon, 'header_tag': tag}


def discover(args):
    src = Path(args.input).expanduser().resolve()
    lines, encoding = read_lines(src, args.encoding)
    senders, system_senders, tags = Counter(), Counter(), Counter()
    errors = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            m = parse_message_line(line)
        except ValueError as exc:
            errors.append({'line': n, 'reason': str(exc)})
            continue
        tags[m['header_tag'] or 'text'] += 1
        if m['header_tag'] == '系统消息' or m['content'].startswith('[系统消息]'):
            system_senders[m['sender']] += 1
        else:
            senders[m['sender']] += 1
    result = {'physical_lines': len(lines), 'encoding': encoding,
              'participants': dict(senders), 'system_sender_headers': dict(system_senders),
              'header_types': dict(tags), 'unparsed': errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def prepare(args):
    src = Path(args.input).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f'输入文件不存在: {src}')
    out = Path(args.out).expanduser().resolve()
    if out == src or src.is_relative_to(out):
        raise ValueError('输出目录不能包含输入文件或与输入文件相同')
    if args.target == args.self_name or args.self_name in (args.target_alias or []):
        raise ValueError('A/B 角色映射冲突：本人名称不能同时作为目标人物或别名')
    original_bytes = src.read_bytes()
    source_sha = hashlib.sha256(original_bytes).hexdigest()
    lines, encoding = read_lines(src, args.encoding)
    pieces = original_bytes.decode(encoding).splitlines(keepends=True)
    if args.expect_lines is not None and len(lines) != args.expect_lines:
        raise ValueError(f'行数校验失败：预期 {args.expect_lines}，实际 {len(lines)}。请核对文件。')
    if len(pieces) != len(lines):
        raise ValueError('原始行边界与解码结果不一致，拒绝转换')
    source_id = hashlib.sha256(str(src).encode('utf-8')).hexdigest()[:24]
    aliases = set(args.target_alias or []) | {args.target}
    known = aliases | {args.self_name}
    messages_ex, messages_suhui, system_msgs = [], [], []
    errors, unknowns, blanks, ledger, reversals = [], [], [], [], []
    counts, by_sender, raw_by_sender, headers = Counter(), Counter(), Counter(), Counter()
    dates, exact_groups = [], defaultdict(list)
    prev_ts = None
    for line_no, (raw, piece) in enumerate(zip(lines, pieces), 1):
        raw_exact = piece.rstrip('\r\n') if piece.endswith(('\r', '\n')) else piece
        ending = piece[len(raw_exact):]
        if raw_exact != raw:
            raise ValueError(f'第 {line_no} 行无法保真还原，拒绝转换')
        message_id = hashlib.sha256(f'{source_id}:{line_no}:'.encode() + raw.encode('utf-8')).hexdigest()
        provenance = {'message_id': message_id, 'source_id': source_id,
                      'source_file': str(src), 'source_sha256': source_sha,
                      'source_line': line_no, 'origin': 'real_chat'}
        row = {**provenance, 'raw_line': raw, 'line_ending': ending, 'sequence': line_no}
        exact_groups[raw].append(line_no)
        if not raw.strip():
            row['record_type'] = 'blank'
            blanks.append(row.copy())
            counts['blank'] += 1
            ledger.append(row)
            continue
        try:
            m = parse_message_line(raw, known)
        except ValueError as exc:
            orphan_emoji = bool(BRACKET_GROUP.fullmatch(raw))
            row.update(record_type='quarantine', timestamp=None, sender=None,
                       reason=str(exc), attributable=False,
                       quarantine_kind='orphan_emoji' if orphan_emoji else 'malformed')
            errors.append({'line': line_no, 'reason': str(exc), 'original': raw,
                           'quarantine_kind': row['quarantine_kind'], **provenance})
            ledger.append(row)
            continue
        row.update(m)
        dates.append(m['timestamp'])
        if prev_ts and m['timestamp'] < prev_ts:
            reversals.append({'source_line': line_no, 'previous_timestamp': prev_ts,
                              'timestamp': m['timestamp']})
        prev_ts = m['timestamp']
        headers[m['header_tag'] or 'text'] += 1
        sender, content, tag = m['sender'], m['content'], m['header_tag']
        if tag == '系统消息' or content.startswith('[系统消息]'):
            row['record_type'] = 'system'
            system_msgs.append(row.copy())
            counts['system'] += 1
            ledger.append(row)
            continue
        if sender not in known:
            row['record_type'] = 'unknown_sender'
            unknowns.append({'line': line_no, 'sender': sender, 'original': raw, **provenance})
            ledger.append(row)
            continue
        role = 'A' if sender == args.self_name else 'B'
        name = args.self_name if role == 'A' else args.target
        cat, mtype, kind, for_ex = content_kind(tag, content)
        if tag.startswith('类型') or (tag and tag not in PLACEHOLDER_MARKERS):
            cat, mtype, kind, for_ex = 'unknown_event', 'file', 'placeholder', bool(content.strip())
            if for_ex:
                mtype, kind = 'text', 'text'
        media_present = tag in PLACEHOLDER_MARKERS or bool(PLACEHOLDER_ONLY.fullmatch(content))
        counts['media_placeholder_messages'] += int(media_present)
        counts[cat] += 1
        by_sender[name] += 1
        raw_by_sender[sender] += 1
        row.update(record_type='message', role=role, normalized_sender=name,
                   category=cat, media_placeholder=media_present, indexable=for_ex)
        ledger.append(row)
        messages_suhui.append({
            'ts': m['timestamp'].replace(' ', 'T'), 'sender': role, 'text': content,
            'type': mtype, 'platform': 'chat', 'kind': kind,
            'raw_sender': sender, 'raw_timestamp': m['timestamp'], 'header_tag': tag,
            'event_type': tag or 'text', **provenance,
        })
        if for_ex and content.strip():
            messages_ex.append({'timestamp': m['timestamp'], 'sender': name,
                                'content': content, 'is_target': role == 'B', 'role': role,
                                'raw_sender': sender, 'header_tag': tag, 'event_type': tag or 'text',
                                **provenance})
        else:
            counts['not_indexed_events'] += 1
    duplicate_groups = [{'line_numbers': ns, 'count': len(ns)}
                        for raw, ns in exact_groups.items() if raw.strip() and len(ns) > 1]
    quarantine_allowed = bool(getattr(args, 'allow_quarantine', False))
    only_unattributed = all(e['quarantine_kind'] == 'orphan_emoji' for e in errors)
    permitted = not unknowns and (not errors or (quarantine_allowed and only_unattributed))
    complete = len(messages_suhui) + len(system_msgs) + len(blanks) + len(errors) + len(unknowns) == len(lines)
    if not complete or src.read_bytes() != original_bytes:
        raise ValueError('输入内容变化或有未记账的原始行，拒绝输出')
    report = {
        'schema_version': 2, 'input_file': str(src), 'source_id': source_id,
        'source_sha256': source_sha, 'source_bytes': len(original_bytes), 'encoding_used': encoding,
        'physical_lines': len(lines), 'parsed_user_messages': len(messages_suhui),
        'ex_memory_index_messages': len(messages_ex), 'system_lines_archived': len(system_msgs),
        'blank_lines': len(blanks), 'parse_error_lines': len(errors), 'unknown_sender_lines': len(unknowns),
        'quarantined_unattributed_lines': len(errors), 'quarantine_policy': '保留原文，不推断发送者/时间',
        'media_placeholder_messages': counts['media_placeholder_messages'],
        'media_only_messages': counts['media_only'], 'non_indexed_event_messages': counts['not_indexed_events'],
        'senders': dict(by_sender), 'raw_senders': dict(raw_by_sender),
        'role_mapping': {'A': args.self_name, 'B': args.target},
        'kinds': dict(counts), 'header_types': dict(headers),
        'exact_duplicate_groups': len(duplicate_groups),
        'exact_duplicate_extra_lines': sum(g['count'] - 1 for g in duplicate_groups),
        'duplicate_policy': '全部保留，仅报告疑似重复，不删除真实消息',
        'timestamp_reversals': len(reversals), 'first_timestamp': min(dates) if dates else None,
        'last_timestamp': max(dates) if dates else None, 'target': args.target, 'self': args.self_name,
        'all_lines_accounted': complete, 'original_unchanged': True,
        'strict_parse_passed': not errors and not unknowns,
        'status': ('ready_with_quarantine' if errors else 'ready') if permitted else 'rejected',
        'timezone': 'Asia/Shanghai', 'timezone_basis': '原始时间戳未含时区；按用户指定本地时区解释，不改原始值',
        'source_note': '仅此输入为真实聊天历史；AI会话/模型推断独立保存，禁止回填为真实历史。',
    }
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.chmod(0o700)
    write_json(out / 'audit.json', report)
    write_json(out / 'system_messages.json', system_msgs)
    write_json(out / 'parse_errors.json', errors)
    write_json(out / 'unknown_senders.json', unknowns)
    write_json(out / 'blank_lines.json', blanks)
    write_json(out / 'duplicate_report.json', duplicate_groups)
    write_json(out / 'timestamp_anomalies.json', reversals)
    write_json(out / 'quarantine.json', [r for r in ledger if r['record_type'] == 'quarantine'])
    ledger_path = out / 'source_ledger.jsonl'
    ledger_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ledger), encoding='utf-8')
    ledger_path.chmod(0o600)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not permitted:
        print('严格校验未通过。完整原始行账本及异常已保存；只允许显式隔离无归属表情，不自动修补原文。', file=sys.stderr)
        return 2
    if not messages_ex or not any(m['is_target'] for m in messages_ex):
        print('目标人物没有可导入文字或命名表情。', file=sys.stderr)
        return 2
    write_json(out / 'ex_memory.json', {'schema_version': 2, 'source': report, 'messages': messages_ex})
    write_json(out / 'suhui' / 'messages.json', messages_suhui)
    print(f'结构化转换完成：{len(messages_suhui)} 条明确归属消息；{len(errors)} 行独立隔离；原文件未改变。')
    return 0


def segment_text(text, allowance):
    """摘要提示词的截断策略：保留首尾，避免只保留最早关系阶段。"""
    text = text.strip()
    if len(text) <= allowance:
        return text
    pre = int(allowance * 0.57)
    post = allowance - pre - len('\n\n……（中间部分省略；完整版本仍在本地人物包）……\n\n')
    return text[:pre].rstrip() + '\n\n……（中间部分省略；完整版本仍在本地人物包）……\n\n' + text[-post:].lstrip()


def inject(args):
    ex_skill = Path(args.ex_skill).expanduser().resolve()
    sd = Path(args.suhui_dir).expanduser().resolve()
    if not ex_skill.is_file() or ex_skill.name != 'SKILL.md':
        raise FileNotFoundError(f'ex-memory 的 SKILL.md 不存在: {ex_skill}')
    if not sd.is_dir() or not (sd / 'persona.md').is_file():
        raise FileNotFoundError(f'人物包目录或 persona.md 不存在: {sd}')
    if args.max_chars < 4000:
        raise ValueError('为保留时段人格与记忆，--max-chars 不应小于 4000')
    portions = [('persona.md', 0.55), ('memories.md', 0.35), ('user_profile.md', 0.10)]
    fragments = []
    total_budget = args.max_chars
    for filename, ratio in portions:
        f = sd / filename
        if f.is_file():
            content = f.read_text(encoding='utf-8')
            allowance = int(total_budget * ratio)
            fragments.append(f'### 人物包来源：{filename}\n' + segment_text(content, allowance))
    extra = '''\n## 优先级与真实性边界\n- 此处的关系画像来自历史聊天记录的模型蒸馏，需要和检索得到的真实原话核对；不是对方的真实内心报告。\n- A 是用户，B 是目标人物；不得混淆发言者。\n- 对于关系阶段、时间、共同经历和对方意图，优先遵循真实聊天证据，而不是 AI 编造。\n- 不得假定双方是恋人或处于某个感情阶段，除非真实记录有依据。\n- 不要把新生成的 AI 聊天和未来模拟事件保存成真实的历史经历。\n- 模拟回复可以参考目标人物常用口头禅、断句、表情和特定时期的表达习惯，但不能宣称自己就是现实中的目标人物。\n'''
    block = '\n\n' + START_MARK + '\n\n## 证据约束人物包：阶段人格、共同经历与情感表达\n\n' \
            + '\n\n'.join(fragments) + '\n' + extra + '\n' + END_MARK + '\n'
    original = ex_skill.read_text(encoding='utf-8')
    regex = re.compile(r'\n*' + re.escape(START_MARK) + r'.*?' + re.escape(END_MARK) + r'\n*', re.DOTALL)
    if START_MARK in original and END_MARK not in original:
        raise ValueError('发现不完整注入标记，拒绝修改 SKILL.md。请先手动检查。')
    cleaned = regex.sub('\n', original).rstrip() + '\n'
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    skill_bak = ex_skill.with_name(f'SKILL.md.bak-{stamp}')
    shutil.copy2(ex_skill, skill_bak)
    ex_skill.write_text(cleaned + block, encoding='utf-8')
    # 将 CLI 导入的历史数据标记成“有来源聊天记录”，避免默认热恋模式擅自补关系。
    meta_path = ex_skill.parent / 'meta.json'
    if meta_path.is_file() and not args.keep_stage:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta_bak = meta_path.with_name(f'meta.json.bak-{stamp}')
        shutil.copy2(meta_path, meta_bak)
        meta['creation_source'] = 'chat_import'
        meta['stage'] = 'source'
        meta['relationship_stage'] = 'source'
        write_json(meta_path, meta)
        print(f'关系基调设为 source（以真实历史证据为准），备份：{meta_bak}')
    print(f'注入成功：{ex_skill}\n已备份：{skill_bak}\n注入字符数：{len(block)}')
    print('请重启 ex-memory 服务或 CLI，使对话引擎重新读取 SKILL.md。')
    return 0


def main():
    ap = argparse.ArgumentParser(description='聊天记录 1 行 1 消息转为 ex-memory 输入，及安全的人格注入')
    sp = ap.add_subparsers(dest='cmd', required=True)
    d = sp.add_parser('discover', help='本地识别发送者和异常行号，不输出聊天正文')
    d.add_argument('--input', required=True)
    d.add_argument('--encoding', default='auto')
    d.set_defaults(func=discover)
    a = sp.add_parser('prepare', help='解析原始 TXT，校验消息边界与 A/B 映射')
    a.add_argument('--input', required=True, help='原始聊天记录 TXT')
    a.add_argument('--target', required=True, help='目标人物在导出中的显示名')
    a.add_argument('--self-name', default='我', help='你在导出文件中的发送者名称，默认 我')
    a.add_argument('--target-alias', action='append', help='若目标人物在普通聊天行中还有其他发言名，可多次设置')
    a.add_argument('--out', required=True, help='结构化结果目录')
    a.add_argument('--expect-lines', type=int, help='预期物理行数（用于核对导出完整性）')
    a.add_argument('--encoding', default='auto', help='auto / utf-8-sig / gb18030')
    a.add_argument('--allow-quarantine', action='store_true', help='仅将无时间/发送者的孤立表情行隔离，不推测归属；其余异常仍拒绝')
    a.set_defaults(func=prepare)
    b = sp.add_parser('inject', help='将证据约束人物包的摘要安全注入 ex-memory 的 SKILL.md')
    b.add_argument('--ex-skill', required=True, help='ex-memory 镜像目录下的 SKILL.md')
    b.add_argument('--suhui-dir', required=True, help='已生成的证据约束人物包目录（兼容选项名）')
    b.add_argument('--max-chars', type=int, default=16000, help='注入人物画像文本的总长度预算，默认 16000 字符')
    b.add_argument('--keep-stage', action='store_true', help='不更改 ex-memory 的关系阶段配置')
    b.set_defaults(func=inject)
    args = ap.parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
