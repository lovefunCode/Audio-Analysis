# -*- coding: utf-8 -*-
import logging
import os

logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO"),
)
logger = logging.getLogger(__file__)
