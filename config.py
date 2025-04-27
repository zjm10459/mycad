from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    CAD_TYPE: str = "AutoCAD.Application.25"

    class Config:
        env_file = (".env", ".env.local")

def get_settings() -> Settings:
    return Settings()

config = get_settings()


