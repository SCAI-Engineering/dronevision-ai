# Reproduce the results

## Install the AI block

```bash
git clone https://github.com/SCAI-Engineering/dronevision-ai.git
cd dronevision-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pi,dev]"
pytest
```

The core package requires Python 3.10 or later. The `pi` extra installs ZeroMQ and ONNX Runtime; the 191-test suite requires no simulator or hardware.

## Run the shipped corpus

=== "Colour reference"

    ```bash
    python -m bench.accuracy
    ```

=== "YOLO / ONNX"

    ```bash
    python -m bench.accuracy \
      --detector yolo --runtime onnx --threads 4
    ```

=== "Occlusion"

    ```bash
    python -m bench.accuracy \
      --occlude cam_ne,cam_sw
    ```

=== "Include JPEG decode"

    ```bash
    python -m bench.accuracy --mode encoded
    ```

The corpus contains 337 synchronized four-camera frame sets, 30 seconds of JPEG quality-90 imagery and time-aligned ground truth. It is approximately 13 MB and remains inspectable as `meta.json`, `manifest.jsonl` and `frames.zip`.

## Benchmark a Raspberry Pi

The board does not need GitHub credentials. From the development machine:

```bash
./bench/pi.sh info
./bench/pi.sh setup
./bench/pi.sh bench
./bench/pi.sh run -m bench.accuracy \
  --detector yolo --runtime onnx --threads 4
```

The helper transfers the committed tree, prepares the environment, runs the sweep and pulls machine-readable JSON back into `bench/out/`.

## Connect to the live simulator

```bash
# In the companion simulator repository
./services.sh up
./drone.sh takeoff

# In dronevision-ai, on a laptop or Arm board
python -m dronevision.service \
  --frames tcp://<simulator-host>:5555 \
  --state <simulator-host>:5601
```

To record another deterministic corpus:

```bash
python -m bench.record --seconds 30
```

## Preview this documentation

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000`. The server rebuilds when Markdown, CSS or configuration files change. Nothing is published by this command.

