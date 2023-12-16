import io
import subprocess as sp
import sys
import select
from typing import Dict, Tuple, Optional, IO
from pathlib import Path

#demucs 관련 변수
model = "htdemucs"
mp3 = False  # WAV 형식으로 저장
float32 = False
int24 = False
two_stems = None  # None으로 설정하면 모든 소스를 분리
in_path = 'content'
def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()


def separate(input_audio_path, output_folder_path):
    cmd = ["python3", "-m", "demucs.separate", "-o", str(output_folder_path), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]
    p = sp.Popen(cmd + [str(input_audio_path)], stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")


def separate_and_save(input_audio_path, output_folder_path):
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(exist_ok=True)
    separate(input_audio_path, output_folder_path)
