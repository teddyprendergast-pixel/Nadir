import sqlite3
import time
import os
from typing import Dict, Any, Optional, List, Tuple

class BiodiversityLogger:
    """SQLite-backed spatial logger for forest plant and wildlife discoveries.
    
    Logs exact 3D coordinates (X, Y, Z), timestamps, confidence scores,
    and metadata for both botanical species (Idea B) and bioacoustic calls (Idea C).
    """
    
    def __init__(self, db_path: str = "forest_biodiversity.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    category TEXT NOT NULL,      -- 'botanical' or 'bioacoustic'
                    label TEXT NOT NULL,         -- Species or sound name
                    confidence REAL NOT NULL,    -- 0.0 to 1.0
                    pos_x REAL NOT NULL,         -- 3D coordinate X (meters)
                    pos_y REAL NOT NULL,         -- 3D coordinate Y (meters)
                    pos_z REAL NOT NULL,         -- 3D coordinate Z (meters)
                    extra_data TEXT             -- JSON string or audio clip path
                )
            """)
            conn.commit()
            
    def log_observation(self, category: str, label: str, confidence: float,
                        position: Tuple[float, float, float], extra_data: str = ""):
        """Insert a new geotagged biodiversity finding."""
        x, y, z = position
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO observations (timestamp, category, label, confidence, pos_x, pos_y, pos_z, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (time.time(), category, label, float(confidence), float(x), float(y), float(z), extra_data))
            conn.commit()
        print(f"🌲 [Biodiversity Logged] [{category.upper()}] {label} ({confidence*100:.1f}%) at pos=({x:.2f}, {y:.2f}, {z:.2f})")
        
    def get_summary(self) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, COUNT(*) FROM observations GROUP BY category")
            return cursor.fetchall()
