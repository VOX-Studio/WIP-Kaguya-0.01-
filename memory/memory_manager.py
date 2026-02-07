"""
Système de mémoire de Kaguya
Trois couches: court terme, long terme, et knowledge base
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
import math


class MemoryType(Enum):
    """Types de mémoire"""
    SHORT_TERM = "short_term"      # Session actuelle
    LONG_TERM = "long_term"        # Préférences, décisions, projets
    KNOWLEDGE = "knowledge"        # Faits appris depuis Wikipédia


class MemoryPriority(Enum):
    """Priorités de mémoire"""
    CRITICAL = 1.0      # Ne jamais oublier
    HIGH = 0.8          # Très important
    MEDIUM = 0.5        # Important
    LOW = 0.3           # Peut être oublié
    TRIVIAL = 0.1       # Déchet temporaire


@dataclass
class MemoryEntry:
    """Entrée de mémoire"""
    id: Optional[int] = None
    timestamp: float = 0.0
    content: str = ""
    memory_type: str = MemoryType.SHORT_TERM.value
    priority: float = MemoryPriority.MEDIUM.value
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    last_accessed: float = 0.0
    access_count: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convertir en dictionnaire pour stockage"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'content': self.content,
            'memory_type': self.memory_type,
            'priority': self.priority,
            'tags': json.dumps(self.tags),
            'metadata': json.dumps(self.metadata),
            'last_accessed': self.last_accessed,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Créer depuis dictionnaire"""
        data['tags'] = json.loads(data.get('tags', '[]'))
        data['metadata'] = json.loads(data.get('metadata', '{}'))
        return cls(**data)


class MemoryDecay:
    """Gestion du decay (oubli progressif) des mémoires"""
    
    @staticmethod
    def calculate_decay(entry: MemoryEntry, decay_factor_days: float = 120.0) -> float:
        """
        Calculer le facteur de decay basé sur le temps et les accès
        
        Returns:
            float: Priority ajustée (0.0 - 1.0)
        """
        age_days = (time.time() - entry.timestamp) / (24 * 3600)
        
        # Decay exponentiel basé sur l'âge
        time_decay = math.exp(-age_days / decay_factor_days)
        
        # Boost basé sur les accès récents
        access_recency = (time.time() - entry.last_accessed) / (24 * 3600)
        access_boost = 1.0 / (1.0 + access_recency / 30.0)  # Boost si accédé récemment
        
        # Boost basé sur la fréquence d'accès
        frequency_boost = min(1.0, entry.access_count / 10.0)  # Max boost à 10 accès
        
        # Priorité finale
        adjusted_priority = entry.priority * time_decay * (1.0 + access_boost * 0.3 + frequency_boost * 0.2)
        
        return min(1.0, max(0.0, adjusted_priority))


class MemoryManager:
    """Gestionnaire principal de la mémoire"""
    
    def __init__(self, memory_dir: str = "./data/memory"):
        """Initialiser le gestionnaire de mémoire"""
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # Bases de données SQLite
        self.short_term_db = os.path.join(memory_dir, "short_term.db")
        self.long_term_db = os.path.join(memory_dir, "long_term.db")
        self.knowledge_db = os.path.join(memory_dir, "knowledge.db")
        
        # Initialiser les tables
        self._init_databases()
    
    def _init_databases(self):
        """Initialiser les tables SQL"""
        schema = """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            priority REAL NOT NULL,
            tags TEXT,
            metadata TEXT,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 0
        );
        
        CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
        CREATE INDEX IF NOT EXISTS idx_priority ON memories(priority);
        CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
        """
        
        for db_path in [self.short_term_db, self.long_term_db, self.knowledge_db]:
            conn = sqlite3.connect(db_path)
            conn.executescript(schema)
            conn.commit()
            conn.close()
    
    def _get_connection(self, memory_type: MemoryType) -> sqlite3.Connection:
        """Obtenir la connexion appropriée selon le type de mémoire"""
        db_map = {
            MemoryType.SHORT_TERM: self.short_term_db,
            MemoryType.LONG_TERM: self.long_term_db,
            MemoryType.KNOWLEDGE: self.knowledge_db
        }
        return sqlite3.connect(db_map[memory_type])
    
    def add(self, entry: MemoryEntry) -> int:
        """Ajouter une entrée de mémoire"""
        memory_type = MemoryType(entry.memory_type)
        conn = self._get_connection(memory_type)
        cursor = conn.cursor()
        
        data = entry.to_dict()
        del data['id']  # Auto-increment
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        
        cursor.execute(
            f"INSERT INTO memories ({columns}) VALUES ({placeholders})",
            list(data.values())
        )
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return entry_id
    
    def get(self, entry_id: int, memory_type: MemoryType) -> Optional[MemoryEntry]:
        """Récupérer une entrée par ID"""
        conn = self._get_connection(memory_type)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            entry = MemoryEntry.from_dict(dict(row))
            # Mettre à jour l'accès
            self._update_access(entry_id, memory_type)
            return entry
        return None
    
    def search(self, 
               query: str = "", 
               memory_type: Optional[MemoryType] = None,
               tags: Optional[List[str]] = None,
               min_priority: float = 0.0,
               limit: int = 10) -> List[MemoryEntry]:
        """Rechercher des entrées de mémoire"""
        results = []
        
        types_to_search = [memory_type] if memory_type else list(MemoryType)
        
        for mtype in types_to_search:
            conn = self._get_connection(mtype)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            sql = "SELECT * FROM memories WHERE priority >= ?"
            params = [min_priority]
            
            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")
            
            sql += " ORDER BY priority DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                entry = MemoryEntry.from_dict(dict(row))
                
                # Filtrer par tags si spécifié
                if tags and not any(tag in entry.tags for tag in tags):
                    continue
                
                results.append(entry)
        
        return results[:limit]
    
    def _update_access(self, entry_id: int, memory_type: MemoryType):
        """Mettre à jour les statistiques d'accès"""
        conn = self._get_connection(memory_type)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (time.time(), entry_id))
        
        conn.commit()
        conn.close()
    
    def apply_decay(self, 
                    memory_type: MemoryType,
                    decay_factor_days: float = 120.0,
                    min_threshold: float = 0.1):
        """Appliquer le decay et supprimer les entrées obsolètes"""
        conn = self._get_connection(memory_type)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories")
        rows = cursor.fetchall()
        
        to_delete = []
        to_update = []
        
        for row in rows:
            entry = MemoryEntry.from_dict(dict(row))
            new_priority = MemoryDecay.calculate_decay(entry, decay_factor_days)
            
            if new_priority < min_threshold:
                to_delete.append(entry.id)
            elif new_priority != entry.priority:
                to_update.append((new_priority, entry.id))
        
        # Supprimer les entrées obsolètes
        if to_delete:
            cursor.executemany("DELETE FROM memories WHERE id = ?", 
                             [(id,) for id in to_delete])
        
        # Mettre à jour les priorités
        if to_update:
            cursor.executemany("UPDATE memories SET priority = ? WHERE id = ?", 
                             to_update)
        
        conn.commit()
        conn.close()
        
        return len(to_delete), len(to_update)
    
    def clear_short_term(self):
        """Vider la mémoire court terme (nouvelle session)"""
        conn = self._get_connection(MemoryType.SHORT_TERM)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories")
        conn.commit()
        conn.close()
    
    def get_context(self, max_entries: int = 20) -> str:
        """Obtenir le contexte récent pour l'agent"""
        # Récupérer les entrées les plus pertinentes
        short_term = self.search(memory_type=MemoryType.SHORT_TERM, limit=10)
        long_term = self.search(memory_type=MemoryType.LONG_TERM, 
                               min_priority=0.5, limit=10)
        
        context_parts = []
        
        if short_term:
            context_parts.append("=== Contexte récent ===")
            for entry in short_term[-5:]:  # 5 dernières
                context_parts.append(f"- {entry.content}")
        
        if long_term:
            context_parts.append("\n=== Informations importantes ===")
            for entry in long_term[:5]:  # 5 plus importantes
                context_parts.append(f"- {entry.content}")
        
        return "\n".join(context_parts)
    
    def stats(self) -> Dict[str, Any]:
        """Statistiques de la mémoire"""
        stats = {}
        
        for mtype in MemoryType:
            conn = self._get_connection(mtype)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(priority) FROM memories")
            avg_priority = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            stats[mtype.value] = {
                'count': count,
                'avg_priority': round(avg_priority, 2)
            }
        
        return stats


if __name__ == "__main__":
    # Test du système de mémoire
    manager = MemoryManager()
    
    # Ajouter quelques entrées
    entry1 = MemoryEntry(
        content="L'utilisateur préfère les réponses courtes",
        memory_type=MemoryType.LONG_TERM.value,
        priority=MemoryPriority.HIGH.value,
        tags=["preferences", "communication"]
    )
    
    entry2 = MemoryEntry(
        content="Discussion sur les jeux vidéo RPG",
        memory_type=MemoryType.SHORT_TERM.value,
        priority=MemoryPriority.MEDIUM.value,
        tags=["conversation", "gaming"]
    )
    
    id1 = manager.add(entry1)
    id2 = manager.add(entry2)
    
    print("Entrées ajoutées:", id1, id2)
    print("\nContexte:", manager.get_context())
    print("\nStats:", json.dumps(manager.stats(), indent=2))
