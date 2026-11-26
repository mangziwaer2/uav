"""Serve the exact public repair table through bounded, unranked read-only pages."""

import json

from .passive_repair_actions import SOURCE_COLUMNS
from .passive_route_exchange import REPAIR_WAYPOINTS, receive_route_request, seal_route_request


INPUT_SECTIONS = ('buildings', 'new_first_targets', 'points', 'routes', REPAIR_WAYPOINTS, 'locked_intervals_s')
PAGE_BYTES = 8192
VIEW_INSTRUCTIONS = (
    'The frozen request is split into numbered sections. Read any section from any zero-based index; '
    'follow next_start until null for the complete section. Inputs and rows have not been ranked or rounded. '
    'Each actions row replaces sources with [source_start, source_count] into sources. Each sources row '
    'replaces edits with [edit_start, edit_count] into edits. Slices are consecutive; zero count is empty. '
    'Use the original request ID and action ID in your reply. A null risk is not zero or illegal.'
)


def _json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


class RepairPacketView:
    """Snapshot one already allowlisted request; expose no filesystem or scoring calls."""

    def __init__(self, wire, *, page_bytes=PAGE_BYTES):
        if type(page_bytes) is not int or page_bytes <= 0:
            raise ValueError('positive page size in bytes required')
        envelope = json.loads(wire)
        if (not isinstance(envelope, dict) or envelope.keys() != {'input', 'instructions'}
                or not isinstance(envelope['input'], dict)):
            raise ValueError('expected a canonical request envelope')
        packet = dict(envelope['input'])
        if any(not isinstance(packet.get(name), list) for name in INPUT_SECTIONS):
            raise ValueError('expected all public repair sections')
        request = seal_route_request(packet, envelope['instructions'])
        if request.wire != wire or request.action_count is None:
            raise ValueError('expected a complete canonical repair request')

        sections = {name: packet.pop(name) for name in INPUT_SECTIONS}
        table = dict(packet.pop('action_table'))
        sections['actions'], sections['sources'], sections['edits'] = self._split(table.pop('rows'))
        self._sha256 = request.request_sha256
        self._sections = {name: tuple(map(_json, rows)) for name, rows in sections.items()}
        self._page_bytes = page_bytes
        self._overheads = {name: max(len(self._encode_page(name, len(rows), len(rows), b'')),
                                     len(self._encode_page(name, len(rows), None, b'')))
                           for name, rows in self._sections.items()}
        self.manifest = _json(dict(version=1, request_sha256=self._sha256, wire_bytes=len(wire),
                                   page_bytes=page_bytes, instructions=envelope['instructions'],
                                   view_instructions=VIEW_INSTRUCTIONS, input=packet, action_table=table,
                                   sections={name: len(rows) for name, rows in self._sections.items()}))
        if len(self.manifest) > page_bytes or any(
                max(map(len, rows), default=0) + self._overheads[name] > page_bytes
                for name, rows in self._sections.items()):
            raise ValueError('a complete packet record cannot fit the configured page size')

    @staticmethod
    def _split(rows):
        actions, sources, edits = [], [], []
        for row in rows:
            start = len(sources)
            if not isinstance(row[-1], list):
                raise ValueError('expected an ordered list of action sources')
            for source in row[-1]:
                if (not isinstance(source, list) or len(source) != len(SOURCE_COLUMNS)
                        or not isinstance(source[1], list)):
                    raise ValueError('invalid source record in the frozen action table')
                base, original_edits, *comparison = source
                span = [len(edits), len(original_edits)]
                edits.extend(original_edits)
                sources.append([base, span, *comparison])
            actions.append([*row[:-1], [start, len(sources) - start]])
        return actions, sources, edits

    def _encode_page(self, section, start, next_start, body):
        metadata = _json(dict(request_sha256=self._sha256, section=section, start=start, next_start=next_start))
        return metadata[:-1] + b',"rows":[' + body + b']}'

    def read_page(self, section, start=0):
        """Read a contiguous page from any index, at most page_bytes including metadata."""
        if not isinstance(section, str) or section not in self._sections or type(start) is not int:
            raise ValueError('unknown section or invalid start index')
        records = self._sections[section]
        if not 0 <= start <= len(records):
            raise ValueError('start index outside section')
        end = start
        available = self._page_bytes - self._overheads[section]
        while end < len(records):
            size = len(records[end]) + int(end > start)
            if size > available:
                break
            available -= size
            end += 1
        return self._encode_page(section, start, end if end < len(records) else None, b','.join(records[start:end]))


def receive_indexed_repair_request(wire, expected_sha256, open_session, *, page_bytes=PAGE_BYTES):
    """After exact-byte verification, give a fresh session only the manifest and page function."""
    def open_verified(exact):
        try:
            view = RepairPacketView(exact, page_bytes=page_bytes)
        except ValueError as error:
            raise OSError('indexed public request failed preflight') from error
        return open_session(view.manifest, view.read_page)

    return receive_route_request(wire, expected_sha256, open_verified)
