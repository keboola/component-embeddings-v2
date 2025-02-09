FROM python:3.12-slim
ENV PYTHONIOENCODING=utf-8

# install gcc to be able to build packages - e.g. required by regex, dateparser, also required for pandas
RUN apt-get update && apt-get install -y build-essential curl

# Download the latest installer
ADD --chmod=755 https://astral.sh/uv/0.5.29/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Add the uv binary to the PATH
ENV PATH=/root/.local/bin:$PATH
# Install the requirements
RUN uv pip install flake8 --system --no-cache

# Install the component
COPY /src /code/src/
COPY /tests /code/tests/
COPY /scripts /code/scripts/
COPY flake8.cfg /code/flake8.cfg
COPY deploy.sh /code/deploy.sh
COPY pyproject.toml /code/pyproject.toml
COPY uv.lock /code/uv.lock

# Install the requirements
RUN uv pip install -r /code/pyproject.toml --system --no-cache

WORKDIR /code/

CMD ["python", "-u", "/code/src/component.py"]