"""隔离 ex-memory 导入副作用；只影响本项目进程，不读取其他软件凭据。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = os.environ.get('CHAT_DATA_DIR')
DATA_ROOT = Path(_DATA_ROOT).expanduser().resolve() if _DATA_ROOT else ROOT
UPSTREAM = ROOT / 'ex-memory'


def prepare_upstream():
    """在导入任何上游模块前调用；模型凭据只由 privacy 的授权配置读取。"""
    for path in (ROOT, UPSTREAM):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    overrides = {
        'EX_MEMORY_USE_KEYCHAIN': 'false', 'EX_MEMORY_LOAD_DOTENV': 'false',
        'LLM_API_KEY': '', 'DEEPSEEK_API_KEY': '', 'EMBEDDING_API_KEY': '',
        'LLM_BASE_URL': 'http://127.0.0.1:9/v1', 'LLM_MODEL': 'unconfigured',
        'EMBEDDING_BASE_URL': 'http://127.0.0.1:9/v1', 'EMBEDDING_MODEL': 'local-bge-small-zh',
        'DATABASE_URL': '', 'REDIS_URL': '', 'VECTOR_BACKEND': 'chroma',
        'OTEL_EXPORTER_OTLP_ENDPOINT': '', 'METRICS_BEARER_TOKEN': '',
        'S3_BUCKET': '', 'REQUIRE_PHONE_VERIFICATION': 'false', 'REQUIRE_AGE_CONFIRMATION': 'false',
        'SINGLE_USER_MODE': 'false', 'LOG_LEVEL': 'WARNING', 'DEBUG': 'false',
        'ANONYMIZED_TELEMETRY': 'False', 'CHROMA_TELEMETRY': 'False',
        'HF_HOME': str(DATA_ROOT / 'data' / 'hf-cache'),
        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'DO_NOT_TRACK': '1',
    }
    os.environ.update(overrides)
    # 上游模块把若干路径固化为全局变量。适配器必须在第一次 import 时及
    # 后续重复初始化时均重指到本安装的数据根，不能写 bundled 源码目录。
    import config
    config.PROJECT_DIR = DATA_ROOT
    config.EXES_DIR = DATA_ROOT / 'exes'
    config.LOG_DIR = DATA_ROOT / 'logs'
    import server.auth
    server.auth.DB_DIR = DATA_ROOT / 'data'
    server.auth.DB_PATH = server.auth.DB_DIR / 'users.db'
    return UPSTREAM


prepare_upstream()
