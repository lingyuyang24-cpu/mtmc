"""Subprocess tests for real worker output handling (no model weights)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from web.worker import redact_error


class WorkerOutputTests(unittest.TestCase):
    def test_replaces_stale_console_and_suppresses_python_and_native_output(self):
        # Keep descriptor changes out of the test runner. Simulate the stale
        # console wrapper that Windows retains when fd 1/2 are redirected.
        script = '''
import json, os, sys
from pathlib import Path
from web.worker import silence_worker_output
class StaleConsole:
    def write(self, value):
        raise OSError(1, 'stale Windows console')
    def flush(self):
        raise OSError(1, 'stale Windows console')
sys.stdout = sys.stderr = StaleConsole()
silence_worker_output()
print('model initialization: 中文')
sys.stdout.flush()
print('private decoder URL', file=sys.stderr)
sys.stderr.flush()
os.write(1, b'native stdout')
os.write(2, b'native stderr')
Path(sys.argv[1]).write_text(json.dumps({'initialized': True, 'closed': sys.stdout.closed}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / 'result.json'
            result = subprocess.run([sys.executable, '-B', '-c', script, str(result_path)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertEqual(result.stdout, b'')
            self.assertEqual(result.stderr, b'')
            self.assertEqual(json.loads(result_path.read_text()), {'initialized': True, 'closed': False})

    def test_error_diagnostics_redact_sources_and_stream_credentials(self):
        source = 'rtsp://user:password@camera/live'
        message = 'failed ' + source + '\nredirect https://other:secret@host/stream'
        redacted = redact_error(message, [source])
        self.assertNotIn('password', redacted)
        self.assertNotIn('secret', redacted)
        self.assertNotIn('rtsp://', redacted)
        self.assertNotIn('https://', redacted)


if __name__ == '__main__':
    unittest.main()
