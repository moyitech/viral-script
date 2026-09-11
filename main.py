"""Launch the HyScript desktop application from the project root."""

import logging

from app.desktop.__main__ import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
