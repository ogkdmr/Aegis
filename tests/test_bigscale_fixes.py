"""Tests for the two bigscale fixes:
  1. _popen_with_retry — retries on EAGAIN instead of crashing.
  2. instance.sh.j2 template — includes the tiktoken cache pre-warm block.
"""
import errno
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Fix 1: _popen_with_retry
# ---------------------------------------------------------------------------

from aegis.launcher import _popen_with_retry, _POPEN_MAX_RETRIES, _POPEN_RETRY_BASE


class TestPopenWithRetry:
    """_popen_with_retry should absorb EAGAIN and succeed on a later attempt."""

    def _eagain(self):
        err = BlockingIOError()
        err.errno = errno.EAGAIN
        return err

    def test_succeeds_on_first_try(self):
        proc = MagicMock(spec=subprocess.Popen)
        with patch("aegis.launcher.subprocess.Popen", return_value=proc) as mock_popen:
            result = _popen_with_retry(["echo", "hi"])
        mock_popen.assert_called_once()
        assert result is proc

    def test_retries_on_eagain_then_succeeds(self):
        proc = MagicMock(spec=subprocess.Popen)
        side_effects = [self._eagain(), self._eagain(), proc]
        with patch("aegis.launcher.subprocess.Popen", side_effect=side_effects) as mock_popen, \
             patch("aegis.launcher.time.sleep") as mock_sleep:
            result = _popen_with_retry(["echo", "hi"])

        assert mock_popen.call_count == 3
        # Two sleeps before the successful attempt
        assert mock_sleep.call_count == 2
        # Sleep durations double each time
        assert mock_sleep.call_args_list[0] == call(_POPEN_RETRY_BASE * 1)
        assert mock_sleep.call_args_list[1] == call(_POPEN_RETRY_BASE * 2)
        assert result is proc

    def test_reraises_eagain_after_max_retries(self):
        with patch("aegis.launcher.subprocess.Popen", side_effect=self._eagain()), \
             patch("aegis.launcher.time.sleep"):
            with pytest.raises(BlockingIOError) as exc_info:
                _popen_with_retry(["echo", "hi"])
        assert exc_info.value.errno == errno.EAGAIN

    def test_non_eagain_oserror_reraises_immediately(self):
        # ENFILE (too many open files in system) is an OSError that must not
        # be retried — only EAGAIN (fork limit) gets the backoff treatment.
        err = OSError(errno.ENFILE, "too many open files")
        with patch("aegis.launcher.subprocess.Popen", side_effect=err), \
             patch("aegis.launcher.time.sleep") as mock_sleep:
            with pytest.raises(OSError) as exc_info:
                _popen_with_retry(["echo", "hi"])
        assert exc_info.value.errno == errno.ENFILE
        mock_sleep.assert_not_called()

    def test_other_exceptions_reraise_immediately(self):
        with patch("aegis.launcher.subprocess.Popen", side_effect=OSError("nope")), \
             patch("aegis.launcher.time.sleep") as mock_sleep:
            with pytest.raises(OSError, match="nope"):
                _popen_with_retry(["echo", "hi"])
        mock_sleep.assert_not_called()

    def test_max_retries_count_is_reasonable(self):
        # Enough retries to outlast a transient spike (>3) but not forever.
        assert 4 <= _POPEN_MAX_RETRIES <= 16

    def test_total_wait_covers_meaningful_backoff(self):
        # Sum of all wait intervals should be at least a few seconds so the
        # system has time to free process slots between fork attempts.
        total = sum(_POPEN_RETRY_BASE * (2 ** i) for i in range(_POPEN_MAX_RETRIES - 1))
        assert total >= 5.0, f"total backoff {total:.1f}s is too short to be useful"


# ---------------------------------------------------------------------------
# Fix 2: tiktoken cache pre-warm in instance.sh.j2
# ---------------------------------------------------------------------------

def _render_template(extra_kwargs=None):
    """Render the instance.sh.j2 template with minimal required variables."""
    from jinja2 import Environment, PackageLoader
    env = Environment(loader=PackageLoader("aegis", "templates"))
    template = env.get_template("instance.sh.j2")
    kwargs = dict(
        model="openai/gpt-oss-120b",
        tensor_parallel_size=4,
        port=8000,
        hf_home="/tmp/hf_home",
        extra_vllm_args=["--max-model-len", "131072"],
        conda_env="/home/user/assets/my_env.tar.gz",
        apptainer_image=None,
        ze_affinity_mask="0,1,2,3",
        modelinfo_cache_script="/tmp/dummy.py",
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return template.render(**kwargs)


class TestTiktokenPreWarm:
    """The rendered script must include the tiktoken cache pre-warm block."""

    def test_tiktoken_cache_dir_exported(self):
        script = _render_template()
        assert "export TIKTOKEN_CACHE_DIR=/tmp/tiktoken_cache" in script

    def test_flock_block_present(self):
        script = _render_template()
        assert "flock 200" in script
        assert "/tmp/tiktoken_cache.lock" in script

    def test_setup_complete_sentinel_checked(self):
        script = _render_template()
        assert "/tmp/tiktoken_cache/.setup_complete" in script

    def test_harmony_encoding_loaded(self):
        script = _render_template()
        assert "load_harmony_encoding" in script
        assert "HARMONY_GPT_OSS" in script

    def test_pre_warm_runs_before_vllm_serve(self):
        script = _render_template()
        tiktoken_pos = script.index("TIKTOKEN_CACHE_DIR")
        vllm_pos = script.index("vllm serve")
        assert tiktoken_pos < vllm_pos, \
            "tiktoken pre-warm must appear before 'vllm serve'"

    def test_pre_warm_runs_after_conda_unpack(self):
        script = _render_template()
        unpack_pos = script.index("conda-unpack")
        tiktoken_pos = script.index("TIKTOKEN_CACHE_DIR")
        assert unpack_pos < tiktoken_pos, \
            "tiktoken pre-warm must appear after 'conda-unpack'"

    def test_pre_warm_absent_without_conda_env(self):
        """Tiktoken block is inside the conda_env branch; skip for apptainer."""
        script = _render_template(extra_kwargs={"conda_env": None, "apptainer_image": "/img.sif"})
        assert "TIKTOKEN_CACHE_DIR" not in script

    def test_pre_warm_failure_is_non_fatal(self):
        """The block must use '|| true' so a missing harmony package doesn't abort launch."""
        script = _render_template()
        assert "|| true" in script

    def test_flock_fd_does_not_conflict_with_conda_lock(self):
        script = _render_template()
        # conda lock uses fd 9; tiktoken lock must use a different fd
        tiktoken_block_start = script.index("TIKTOKEN_CACHE_DIR")
        tiktoken_snippet = script[tiktoken_block_start: tiktoken_block_start + 400]
        assert "flock 9" not in tiktoken_snippet, \
            "tiktoken lock fd must not collide with conda lock fd (9)"
