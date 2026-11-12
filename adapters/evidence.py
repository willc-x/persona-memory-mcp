"""运行前回查不可变真实历史；哈希/行号/角色/日期/原话不是模型自证。"""
from datetime import datetime, time, timezone
import json
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo('Asia/Shanghai')


def parse_instant(value, end=False):
    if not isinstance(value, str):
        raise ValueError('证据缺少日期')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if len(value) == 10 and end:
        parsed = datetime.combine(parsed.date(), time.max)
    return parsed.replace(tzinfo=parsed.tzinfo or SHANGHAI).astimezone(timezone.utc)


def canonicalize_claim_interval(verifier, claim):
    """只按被引用原话的实际时间收紧claim区间；不推断事件发生日。"""
    if not isinstance(claim, dict) or not isinstance(claim.get('evidence'), list) or not claim['evidence']:
        return claim
    times = []
    for item in claim['evidence']:
        if not isinstance(item, dict) or item.get('sender') not in {'A', 'B'} or not item.get('quote'):
            return claim
        rows = [row for row in verifier.records_at(item.get('source_line'), sender=item['sender'])
                if item['quote'] in (row['content'] or '')]
        if len(rows) != 1:
            return claim
        times.append(rows[0]['timestamp'])
    normalized = dict(claim)
    model_interval = (claim.get('time_start'), claim.get('time_end'))
    normalized['time_start'] = min(times)
    normalized['time_end'] = max(times)
    normalized['time_semantics'] = 'chat_evidence_interval_only_not_event_occurrence_date'
    normalized['time_provenance'] = 'derived_from_exactly_cited_source_timestamps'
    if model_interval != (normalized['time_start'], normalized['time_end']):
        normalized['model_time_interval_before_canonicalization'] = list(model_interval)
    return normalized


class CanonicalVerifier:
    def __init__(self, history):
        self.history = history

    def records_at(self, source_line, *, sender=None, timestamp=None, source_id=None):
        if not isinstance(source_line, int) or isinstance(source_line, bool) or source_line < 1:
            return []
        query = "SELECT r.*,s.source_file FROM records r JOIN sources s USING(source_id) WHERE r.source_line=? AND r.record_type='message'"
        arguments = [source_line]
        if sender is not None:
            query += ' AND r.role=?'
            arguments.append(sender)
        if source_id is not None:
            query += ' AND r.source_id=?'
            arguments.append(source_id)
        with self.history.connect() as db:
            candidates = [dict(row) for row in db.execute(query, arguments)]
        if timestamp is not None:
            try:
                instant = parse_instant(timestamp)
                candidates = [row for row in candidates if parse_instant(row['timestamp']) == instant]
            except (TypeError, ValueError):
                return []
        return candidates

    def verify(self, claim):
        """返回全证据核验并补全来源的claim；任一证据缺失/含混则拒绝。"""
        if not isinstance(claim, dict) or claim.get('category') not in {'fact', 'style', 'emotion_expression', 'relationship', 'inference', 'unknown'}:
            return None
        try:
            lower = parse_instant(claim.get('time_start'))
            upper = parse_instant(claim.get('time_end') or claim.get('time_start'), end=True)
        except (TypeError, ValueError):
            return None
        if upper < lower or not isinstance(claim.get('evidence'), list) or not claim['evidence']:
            return None
        evidence = []
        for item in claim['evidence']:
            if not isinstance(item, dict) or item.get('sender') not in {'A', 'B'} or not isinstance(item.get('quote'), str) or not item['quote']:
                return None
            matches = self.records_at(item.get('source_line'), sender=item['sender'],
                       timestamp=item.get('time') or item.get('timestamp'), source_id=item.get('source_id'))
            matches = [row for row in matches if lower <= parse_instant(row['timestamp']) <= upper
                       and item['quote'] in (row['content'] or '')]
            if len(matches) != 1:
                return None
            row = matches[0]
            if item.get('message_id') and item['message_id'] != row['message_id']:
                return None
            evidence.append({**item, 'excerpt': item['quote'], 'quote': row['content'],
                   'time': row['timestamp'], 'timestamp': row['timestamp'], 'raw_sender': row['sender'],
                   'source_line': row['source_line'], 'source_id': row['source_id'], 'source_file': row['source_file'],
                   'message_id': row['message_id'], 'origin': 'real_chat', 'canonical_verified': True})
        # 这里只认证原话来源。statement的语义概括/归因仍需人工审查，不宣称自动证明心理。
        return {**claim, 'evidence': evidence, 'canonical_verified': True,
                'verification_scope': 'exact_quote_sender_timestamp_source_only'}

    def enrich_sources(self, sources):
        verified = []
        for item in sources:
            if not isinstance(item, dict) or item.get('sender') not in {'A', 'B'}:
                continue
            matches = self.records_at(item.get('source_line'), sender=item['sender'],
                                      timestamp=item.get('time') if not item.get('claim_id') else None,
                                      source_id=item.get('source_id'))
            if len(matches) != 1:
                continue
            row = matches[0]
            verified.append({**item, 'raw_quote': row['content'], 'source_file': row['source_file'],
                   'source_id': row['source_id'], 'source_line': row['source_line'], 'message_id': row['message_id'],
                   'timestamp': row['timestamp'], 'raw_sender': row['sender'], 'origin': 'real_chat',
                   'canonical_verified': True})
        return verified
