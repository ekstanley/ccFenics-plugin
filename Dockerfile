FROM dolfinx/dolfinx:stable

WORKDIR /app

# Install Python dependencies not provided by base image
RUN pip install --no-cache-dir \
    "mcp[cli]>=1.2.0" \
    "pydantic>=2.0" \
    pyvista \
    matplotlib

# Copy source
COPY pyproject.toml .
COPY src/ src/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Copy tests (for in-container testing)
COPY tests/ tests/

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcpuser && \
    mkdir -p /workspace && \
    chown -R mcpuser:mcpuser /app /workspace

USER mcpuser

# Working directory for simulation output
WORKDIR /workspace

ENTRYPOINT ["python", "-m", "dolfinx_mcp"]
