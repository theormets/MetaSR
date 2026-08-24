FROM python:3.10

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget curl git build-essential \
    libopenblas-dev libgmp-dev libmpfr-dev \
    libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────
# Install BEFORE copying project so this layer is cached on code changes
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────
COPY . .

# ── Precompile Julia / PySR at BUILD time ─────────────────────────────
# pysr 1.5+ requires maxsize >= 7  (was 6 before → ValueError)
# juliapkg will download and manage Julia automatically — no manual install needed
RUN printf 'import numpy as np\nfrom pysr import PySRRegressor\nprint("Precompiling Julia...")\nm = PySRRegressor(niterations=1,populations=1,maxsize=7,procs=0,multithreading=False,verbosity=0,random_state=0)\nrng = np.random.default_rng(0)\nm.fit(rng.standard_normal((12,2)),rng.standard_normal(12))\nprint("Julia precompile done.")\n' > /tmp/precompile.py && python3 /tmp/precompile.py

EXPOSE 7860
CMD ["python", "app.py"]