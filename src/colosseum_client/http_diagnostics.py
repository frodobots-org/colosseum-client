"""HTTP diagnostics without response bodies, request headers or signed URLs."""
from http import HTTPStatus
import json
from pathlib import Path
import re
import tempfile


def log_http_error(response, *, operation, log_dir=None):
    record = {'operation': operation, 'status': response.status_code}
    try:
        record['error'] = HTTPStatus(response.status_code).phrase
    except ValueError:
        record['error'] = 'HTTP request failed'
    # Do not serialize arbitrary headers, e.g. Location, Set-Cookie or Authorization.
    for header in ('x-request-id', 'x-amz-request-id'):
        value = response.headers.get(header, '')
        if re.fullmatch(r'[A-Za-z0-9._:+/=-]{1,200}', value):
            record[header] = value
    text = json.dumps(record, sort_keys=True)
    print(text, flush=True)
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            # Unique, private files also avoid overwriting errors within one second.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    prefix='http-error-', suffix='.json', dir=log_dir, delete=False) as output:
                output.write(text + '\n')
        except OSError:
            print('Could not save HTTP diagnostic; see the summary above.', flush=True)
    return record['error']
