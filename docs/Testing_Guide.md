# Testing Guide - Transcription Service Backend

## Overview

This guide covers the comprehensive testing strategy for the Transcription Service Backend, including unit tests, integration tests, and end-to-end tests.

## Testing Pyramid

Our testing strategy follows the testing pyramid principle:

```
        /\
       /  \
      / E2E \
     /______\
    /        \
   /Integration\
  /_____________\
 /              \
/   Unit Tests   \
/__________________\
```

- **Unit Tests (70%)**: Fast, isolated tests for individual components
- **Integration Tests (20%)**: Tests for component interactions
- **End-to-End Tests (10%)**: Full system tests

## Test Structure

### Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests
│   ├── domain/                 # Domain model tests
│   ├── services/               # Service layer tests
│   └── infrastructure/         # Infrastructure tests
├── integration/                # Integration tests
│   ├── test_api_endpoints.py   # API integration tests
│   └── test_transcription_workflow.py
└── e2e/                        # End-to-end tests
    └── test_full_transcription_flow.py
```

### Test Configuration

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
    -ra
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    external: Tests requiring external services
```

#### Coverage Configuration (.coveragerc)
```ini
[run]
source = app
omit = 
    */tests/*
    */venv/*
    */migrations/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Unit Tests

Unit tests focus on testing individual components in isolation.

### Domain Model Tests

Test domain models and business logic:

```python
# Example: Testing TranscriptionJob domain model
def test_transcription_job_status_update():
    job = TranscriptionJob(...)
    job.update_status(JobStatus.PROCESSING)
    
    assert job.status == JobStatus.PROCESSING
    assert job.started_at is not None
    assert job.updated_at is not None
```

### Service Layer Tests

Test business logic and service interactions:

```python
# Example: Testing TranscriptionService
@pytest.mark.unit
async def test_create_transcription_job_success(
    transcription_service,
    mock_repositories,
    sample_user
):
    # Test service logic with mocked dependencies
    job = await transcription_service.create_transcription_job(
        user_id=sample_user.id,
        filename="test.wav",
        file_content=b"audio_data"
    )
    
    assert job.user_id == sample_user.id
    assert job.status == JobStatus.PENDING
```

### Infrastructure Tests

Test infrastructure components:

```python
# Example: Testing Redis client
@pytest.mark.unit
async def test_redis_client_set_get():
    client = RedisClient()
    await client.set("test_key", "test_value")
    value = await client.get("test_key")
    
    assert value == "test_value"
```

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/domain/test_transcription_models.py -v

# Run tests matching pattern
pytest tests/unit/ -k "test_transcription" -v
```

## Integration Tests

Integration tests verify that different components work together correctly.

### Database Integration Tests

Test database operations and repository implementations:

```python
@pytest.mark.integration
async def test_job_repository_crud_operations(test_db, sample_user):
    repo = SQLTranscriptionJobRepository(test_db)
    
    # Create
    job = TranscriptionJob(...)
    await repo.save(job)
    
    # Read
    retrieved_job = await repo.get_by_id(job.id)
    assert retrieved_job is not None
    
    # Update
    retrieved_job.update_status(JobStatus.COMPLETED)
    await repo.save(retrieved_job)
    
    # Delete
    await repo.delete(job.id)
    deleted_job = await repo.get_by_id(job.id)
    assert deleted_job is None
```

### API Integration Tests

Test API endpoints with real database and dependencies:

```python
@pytest.mark.integration
def test_create_job_endpoint(client, test_db, sample_user, sample_audio_file):
    test_db.add(sample_user)
    test_db.commit()
    
    with open(sample_audio_file, 'rb') as audio_file:
        response = client.post(
            f"/api/v1/users/{sample_user.id}/jobs",
            files={"audio_file": ("test.wav", audio_file, "audio/wav")}
        )
    
    assert response.status_code == 201
    job_data = response.json()
    assert job_data["status"] == JobStatus.PENDING.value
```

### Service Integration Tests

Test service layer with real dependencies:

```python
@pytest.mark.integration
async def test_transcription_workflow_integration(
    test_db,
    file_manager,
    redis_client,
    sample_user,
    sample_audio_file
):
    # Test complete workflow with real components
    service = TranscriptionService(
        job_repository=SQLTranscriptionJobRepository(test_db),
        user_repository=SQLUserRepository(test_db),
        file_manager=file_manager
    )
    
    # Create job
    with open(sample_audio_file, 'rb') as audio_file:
        job = await service.create_transcription_job(
            user_id=sample_user.id,
            filename="test.wav",
            file_content=audio_file
        )
    
    # Verify job was created and file stored
    assert job.status == JobStatus.PENDING
    assert await file_manager.file_exists(job.audio_file_path)
```

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with real services (requires Docker)
docker-compose -f docker-compose.test.yml run --rm integration-tests

# Run specific integration test
pytest tests/integration/test_api_endpoints.py::TestJobsEndpoints::test_create_job_endpoint -v
```

## End-to-End Tests

E2E tests verify the complete system functionality from user perspective.

### Full Workflow Tests

Test complete transcription workflow:

```python
@pytest.mark.e2e
async def test_complete_transcription_pipeline(
    async_client,
    test_db,
    sample_user,
    sample_audio_file,
    mock_ai_models
):
    # 1. Create user
    test_db.add(sample_user)
    test_db.commit()
    
    # 2. Upload audio file
    with open(sample_audio_file, 'rb') as audio_file:
        response = await async_client.post(
            f"/api/v1/users/{sample_user.id}/jobs",
            files={"audio_file": ("test.wav", audio_file, "audio/wav")}
        )
    
    job_id = response.json()["id"]
    
    # 3. Process job (mocked AI processing)
    # ... processing steps ...
    
    # 4. Verify final result
    response = await async_client.get(f"/api/v1/jobs/{job_id}")
    final_job = response.json()
    
    assert final_job["status"] == JobStatus.COMPLETED.value
    assert final_job["transcription_result"]["text"] is not None
```

### Performance Tests

Test system performance under load:

```python
@pytest.mark.e2e
@pytest.mark.slow
async def test_concurrent_job_processing(
    async_client,
    test_db,
    sample_user,
    sample_audio_file
):
    # Test processing multiple jobs concurrently
    num_jobs = 10
    tasks = []
    
    for i in range(num_jobs):
        task = create_and_process_job(async_client, sample_user, sample_audio_file, i)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Verify all jobs completed successfully
    assert len(results) == num_jobs
    for result in results:
        assert result["status"] == "completed"
```

### Running E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run with Docker Compose
docker-compose -f docker-compose.test.yml run --rm e2e-tests

# Run without slow tests
pytest tests/e2e/ -v -m "not slow"

# Run specific E2E test
pytest tests/e2e/test_full_transcription_flow.py::TestFullTranscriptionFlow::test_complete_transcription_pipeline -v
```

## Test Data Management

### Fixtures

Common fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def sample_user():
    return User(
        id=uuid.uuid4(),
        username="testuser",
        email="test@example.com"
    )

@pytest.fixture
def sample_audio_file():
    # Create temporary audio file for testing
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        # Generate test audio data
        # ... audio generation logic ...
        yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)
```

### Database Fixtures

Database setup for tests:

```python
@pytest.fixture(scope="function")
def test_db():
    engine = create_engine(TEST_DATABASE_URL, ...)
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
```

### Mocking External Services

Mock AI models and external dependencies:

```python
@pytest.fixture
def mock_ai_models():
    class MockWhisperModel:
        def generate(self, **kwargs):
            return [[1, 2, 3, 4, 5]]  # Mock token IDs
    
    return {
        "whisper_model": MockWhisperModel(),
        "whisper_processor": MockWhisperProcessor(),
        "diarization_pipeline": MockDiarizationPipeline()
    }
```

## Continuous Integration

### GitHub Actions Configuration

Tests run automatically on every push and pull request:

```yaml
# .github/workflows/ci.yml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: poetry install
      - name: Run unit tests
        run: pytest tests/unit/ --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Environment

Integration and E2E tests run with real services:

```yaml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: test_pass
      POSTGRES_USER: test_user
      POSTGRES_DB: test_db
  
  redis:
    image: redis:7
```

## Performance Testing

### Load Testing with Locust

Create performance tests for API endpoints:

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class TranscriptionServiceUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login or setup user
        pass
    
    @task(3)
    def get_health(self):
        self.client.get("/health")
    
    @task(1)
    def upload_audio(self):
        with open("test_audio.wav", "rb") as f:
            files = {"audio_file": ("test.wav", f, "audio/wav")}
            self.client.post(f"/api/v1/users/{self.user_id}/jobs", files=files)
```

Run performance tests:

```bash
# Run load test
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Run headless load test
locust -f tests/performance/locustfile.py --host=http://localhost:8000 \
       --users=50 --spawn-rate=5 --run-time=5m --headless
```

### Memory and CPU Profiling

Profile application performance:

```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import time

@pytest.mark.performance
def test_memory_usage_during_processing():
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    # ... transcription processing ...
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Assert memory usage is within acceptable limits
    assert memory_increase < 500 * 1024 * 1024  # 500MB max increase
```

## Test Best Practices

### Writing Good Tests

1. **Arrange, Act, Assert**: Structure tests clearly
2. **Single Responsibility**: One test, one concept
3. **Descriptive Names**: Test name explains what is being tested
4. **Independent Tests**: Tests should not depend on each other
5. **Fast Execution**: Keep unit tests fast (< 100ms each)

### Test Data

1. **Use Factories**: Create test data with factory functions
2. **Minimal Data**: Use only necessary data for the test
3. **Avoid Hard-coding**: Use constants or fixtures
4. **Clean Up**: Always clean up test data

### Mocking Guidelines

1. **Mock External Dependencies**: Database, APIs, file system
2. **Don't Mock What You Own**: Test your own code, mock others
3. **Verify Interactions**: Assert that mocks were called correctly
4. **Keep Mocks Simple**: Don't over-complicate mock behavior

### Error Testing

1. **Test Error Paths**: Verify error handling
2. **Test Edge Cases**: Boundary conditions and limits
3. **Test Validation**: Input validation and error messages
4. **Test Recovery**: System behavior after errors

## Debugging Tests

### Debug Failed Tests

```bash
# Run tests with verbose output
pytest tests/unit/test_failing.py -v -s

# Drop into debugger on failure
pytest tests/unit/test_failing.py --pdb

# Run only failed tests from last run
pytest --lf

# Run specific test with extra output
pytest tests/unit/test_specific.py::test_function -v -s --tb=long
```

### Debug Test Environment

```bash
# Check test database
docker-compose -f docker-compose.test.yml exec db-test psql -U test_user -d test_db

# Check test logs
docker-compose -f docker-compose.test.yml logs app-test

# Interactive debugging
docker-compose -f docker-compose.test.yml run --rm app-test bash
```

## Test Metrics and Reporting

### Coverage Reports

Generate coverage reports:

```bash
# HTML coverage report
pytest --cov=app --cov-report=html tests/

# Terminal coverage report
pytest --cov=app --cov-report=term-missing tests/

# XML coverage report (for CI)
pytest --cov=app --cov-report=xml tests/
```

### Test Results

Export test results for CI:

```bash
# JUnit XML for CI integration
pytest --junitxml=test-results.xml tests/

# JSON report
pytest --json-report --json-report-file=test-report.json tests/
```

### Performance Metrics

Track test performance:

```bash
# Show slowest tests
pytest --durations=10 tests/

# Profile test execution
pytest --profile tests/
```

## Test Automation

### Pre-commit Hooks

Run tests before commits:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: pytest tests/unit/
        language: system
        pass_filenames: false
        always_run: true
```

### IDE Integration

### VS Code Configuration

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

## Troubleshooting

### Common Issues

#### Test Database Issues
- Ensure test database is clean before each test
- Check database connection parameters
- Verify migrations are applied

#### Fixture Issues
- Check fixture scope (function vs session)
- Verify fixture dependencies
- Ensure fixtures are properly cleaned up

#### Async Test Issues
- Use `pytest-asyncio` for async tests
- Ensure event loop is properly handled
- Check for async/await usage

#### Mock Issues
- Verify mock targets are correct
- Check mock return values
- Ensure mocks are reset between tests