def main() -> int:
    from .app import main as app_main

    return app_main()

__all__ = ["main"]
