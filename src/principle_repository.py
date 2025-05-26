#!/usr/bin/env python3
"""
Principle Repository

This module provides the PrincipleRepository class, which is responsible for
all CRUD operations for principles in the database. It abstracts the database
interactions and provides a clean interface for working with principles.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleRepository")


class PrincipleRepository:
    """
    Repository class for managing principles in the database.
    
    This class abstracts all database operations related to principles, providing
    methods for creating, reading, updating, and deleting principles and related data.
    """
    
    def __init__(self, db_path: str) -> None:
        """
        Initialize the repository with a database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a new database connection.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable row access by column name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn
    
    def _init_db(self) -> None:
        """Initialize the database if it doesn't exist."""
        try:
            # Check if database has the required tables
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='principles'")
            if cursor.fetchone() is None:
                logger.info(f"Initializing database at {self.db_path}")
                with open('src/principle_database_schema.sql', 'r') as f:
                    schema_sql = f.read()
                conn.executescript(schema_sql)
                conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _record_audit(
        self, 
        conn: sqlite3.Connection, 
        table_name: str, 
        record_id: int, 
        action: str, 
        old_values: Optional[Dict[str, Any]] = None, 
        new_values: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """
        Record an action in the audit log.
        
        Args:
            conn: Database connection
            table_name: Name of the table being modified
            record_id: ID of the record being modified
            action: Action type ('INSERT', 'UPDATE', 'DELETE')
            old_values: Previous values (for UPDATE/DELETE)
            new_values: New values (for INSERT/UPDATE)
            user_id: ID of the user making the change
        """
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_log (table_name, record_id, action, old_values, new_values, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                table_name,
                record_id,
                action,
                json.dumps(old_values) if old_values else None,
                json.dumps(new_values) if new_values else None,
                user_id
            )
        )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Principle Category Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all principle categories.
        
        Returns:
            List of category dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, description, parent_category_id, created_at, updated_at "
            "FROM principle_categories "
            "ORDER BY name"
        )
        categories = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return categories
    
    def get_category(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a principle category by ID.
        
        Args:
            category_id: Category ID
            
        Returns:
            Category dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, description, parent_category_id, created_at, updated_at "
            "FROM principle_categories "
            "WHERE id = ?",
            (category_id,)
        )
        category = cursor.fetchone()
        conn.close()
        
        if category:
            return dict(category)
        return None
    
    def create_category(
        self, 
        name: str, 
        description: Optional[str] = None, 
        parent_category_id: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Create a new principle category.
        
        Args:
            name: Category name
            description: Optional description
            parent_category_id: Optional parent category ID
            user_id: Optional ID of the user creating the category
            
        Returns:
            ID of the created category
            
        Raises:
            ValueError: If a category with the same name already exists
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if category with the same name exists
            cursor.execute("SELECT id FROM principle_categories WHERE name = ?", (name,))
            if cursor.fetchone():
                raise ValueError(f"Category with name '{name}' already exists")
            
            cursor.execute(
                "INSERT INTO principle_categories (name, description, parent_category_id, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (name, description, parent_category_id)
            )
            category_id = cursor.lastrowid
            
            # Record audit
            self._record_audit(
                conn=conn,
                table_name="principle_categories",
                record_id=category_id,
                action="INSERT",
                new_values={
                    "name": name,
                    "description": description,
                    "parent_category_id": parent_category_id
                },
                user_id=user_id
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Created principle category: {name} (ID: {category_id})")
            return category_id
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error creating principle category: {str(e)}")
            raise
    
    def update_category(
        self, 
        category_id: int, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        parent_category_id: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update a principle category.
        
        Args:
            category_id: ID of the category to update
            name: Optional new name
            description: Optional new description
            parent_category_id: Optional new parent category ID
            user_id: Optional ID of the user updating the category
            
        Returns:
            True if the category was updated, False if not found
            
        Raises:
            ValueError: If a different category with the same name already exists
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute(
                "SELECT name, description, parent_category_id FROM principle_categories WHERE id = ?",
                (category_id,)
            )
            current = cursor.fetchone()
            if not current:
                conn.close()
                return False
            
            current_dict = dict(current)
            
            # Check if name is being changed and if a different category with the same name exists
            if name and name != current_dict["name"]:
                cursor.execute(
                    "SELECT id FROM principle_categories WHERE name = ? AND id <> ?",
                    (name, category_id)
                )
                if cursor.fetchone():
                    raise ValueError(f"Different category with name '{name}' already exists")
            
            # Build update parts
            update_parts = []
            params = []
            
            if name is not None:
                update_parts.append("name = ?")
                params.append(name)
            
            if description is not None:
                update_parts.append("description = ?")
                params.append(description)
            
            if parent_category_id is not None:
                update_parts.append("parent_category_id = ?")
                params.append(parent_category_id)
            
            # Only update if there are changes
            if update_parts:
                update_parts.append("updated_at = CURRENT_TIMESTAMP")
                query = f"UPDATE principle_categories SET {', '.join(update_parts)} WHERE id = ?"
                params.append(category_id)
                
                cursor.execute(query, params)
                
                # Record audit
                self._record_audit(
                    conn=conn,
                    table_name="principle_categories",
                    record_id=category_id,
                    action="UPDATE",
                    old_values=current_dict,
                    new_values={
                        "name": name if name is not None else current_dict["name"],
                        "description": description if description is not None else current_dict["description"],
                        "parent_category_id": parent_category_id if parent_category_id is not None else current_dict["parent_category_id"]
                    },
                    user_id=user_id
                )
                
                conn.commit()
                logger.info(f"Updated principle category ID {category_id}")
            
            conn.close()
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error updating principle category: {str(e)}")
            raise
    
    def delete_category(self, category_id: int, user_id: Optional[str] = None) -> bool:
        """
        Delete a principle category.
        
        Args:
            category_id: ID of the category to delete
            user_id: Optional ID of the user deleting the category
            
        Returns:
            True if the category was deleted, False if not found
            
        Raises:
            sqlite3.IntegrityError: If the category has principles or is referenced by other categories
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute(
                "SELECT name, description, parent_category_id FROM principle_categories WHERE id = ?",
                (category_id,)
            )
            current = cursor.fetchone()
            if not current:
                conn.close()
                return False
            
            # Check if category has principles
            cursor.execute("SELECT COUNT(*) FROM principles WHERE category_id = ?", (category_id,))
            if cursor.fetchone()[0] > 0:
                raise sqlite3.IntegrityError("Cannot delete category with associated principles")
            
            # Check if category is referenced by other categories
            cursor.execute(
                "SELECT COUNT(*) FROM principle_categories WHERE parent_category_id = ?",
                (category_id,)
            )
            if cursor.fetchone()[0] > 0:
                raise sqlite3.IntegrityError("Cannot delete category referenced by other categories")
            
            # Record audit before deletion
            self._record_audit(
                conn=conn,
                table_name="principle_categories",
                record_id=category_id,
                action="DELETE",
                old_values=dict(current),
                user_id=user_id
            )
            
            # Delete the category
            cursor.execute("DELETE FROM principle_categories WHERE id = ?", (category_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted principle category ID {category_id}")
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error deleting principle category: {str(e)}")
            raise
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Importance Level Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def get_importance_levels(self) -> List[Dict[str, Any]]:
        """
        Get all importance levels.
        
        Returns:
            List of importance level dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, value, description, created_at "
            "FROM importance_levels "
            "ORDER BY value DESC"  # Higher values first
        )
        levels = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return levels
    
    def get_importance_level(self, level_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an importance level by ID.
        
        Args:
            level_id: Importance level ID
            
        Returns:
            Importance level dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, value, description, created_at "
            "FROM importance_levels "
            "WHERE id = ?",
            (level_id,)
        )
        level = cursor.fetchone()
        conn.close()
        
        if level:
            return dict(level)
        return None
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Principle Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def get_principles(
        self,
        category_id: Optional[int] = None,
        importance_level_id: Optional[int] = None,
        tags: Optional[List[str]] = None,
        is_active: Optional[bool] = True
    ) -> List[Dict[str, Any]]:
        """
        Get principles with optional filters.
        
        Args:
            category_id: Optional category ID filter
            importance_level_id: Optional importance level ID filter
            tags: Optional list of tag names to filter by
            is_active: Optional active status filter
            
        Returns:
            List of principle dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT 
                p.id, p.name, p.description, p.category_id, p.importance_level_id,
                p.short_name, p.version, p.is_active, p.created_at, p.updated_at,
                p.created_by, p.updated_by,
                c.name as category_name,
                il.name as importance_level_name,
                il.value as importance_value
            FROM principles p
            JOIN principle_categories c ON p.category_id = c.id
            JOIN importance_levels il ON p.importance_level_id = il.id
        """
        params = []
        where_clauses = []
        
        # Add filters
        if category_id is not None:
            where_clauses.append("p.category_id = ?")
            params.append(category_id)
        
        if importance_level_id is not None:
            where_clauses.append("p.importance_level_id = ?")
            params.append(importance_level_id)
        
        if is_active is not None:
            where_clauses.append("p.is_active = ?")
            params.append(1 if is_active else 0)
        
        # Combine WHERE clauses
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Add ORDER BY
        query += " ORDER BY il.value DESC, p.name"
        
        # Execute query
        cursor.execute(query, params)
        principles = [dict(row) for row in cursor.fetchall()]
        
        # Filter by tags if needed
        if tags:
            # Get principles with specified tags
            tag_placeholders = ", ".join(["?"] * len(tags))
            cursor.execute(f"""
                SELECT DISTINCT p.id
                FROM principles p
                JOIN principle_tags pt ON p.id = pt.principle_id
                JOIN tags t ON pt.tag_id = t.id
                WHERE t.name IN ({tag_placeholders})
            """, tags)
            principle_ids_with_tags = {row["id"] for row in cursor.fetchall()}
            
            # Filter the results
            principles = [p for p in principles if p["id"] in principle_ids_with_tags]
        
        # Get tags for each principle
        for principle in principles:
            cursor.execute("""
                SELECT t.id, t.name
                FROM tags t
                JOIN principle_tags pt ON t.id = pt.tag_id
                WHERE pt.principle_id = ?
                ORDER BY t.name
            """, (principle["id"],))
            principle["tags"] = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return principles
    
    def get_principle(self, principle_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a principle by ID.
        
        Args:
            principle_id: Principle ID
            
        Returns:
            Principle dictionary with related data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get principle
        cursor.execute("""
            SELECT 
                p.id, p.name, p.description, p.category_id, p.importance_level_id,
                p.short_name, p.version, p.is_active, p.created_at, p.updated_at,
                p.created_by, p.updated_by,
                c.name as category_name,
                il.name as importance_level_name,
                il.value as importance_value
            FROM principles p
            JOIN principle_categories c ON p.category_id = c.id
            JOIN importance_levels il ON p.importance_level_id = il.id
            WHERE p.id = ?
        """, (principle_id,))
        
        principle = cursor.fetchone()
        if not principle:
            conn.close()
            return None
        
        principle_dict = dict(principle)
        
        # Get tags
        cursor.execute("""
            SELECT t.id, t.name
            FROM tags t
            JOIN principle_tags pt ON t.id = pt.tag_id
            WHERE pt.principle_id = ?
            ORDER BY t.name
        """, (principle_id,))
        principle_dict["tags"] = [dict(row) for row in cursor.fetchall()]
        
        # Get evaluation criteria
        cursor.execute("""
            SELECT 
                ec.id, ec.type_id, ec.content, ec.parameters, ec.version,
                ec.is_active, ec.created_at, ec.updated_at,
                et.name as type_name, et.requires_llm
            FROM evaluation_criteria ec
            JOIN evaluation_types et ON ec.type_id = et.id
            WHERE ec.principle_id = ?
            ORDER BY ec.updated_at DESC
        """, (principle_id,))
        principle_dict["evaluation_criteria"] = [dict(row) for row in cursor.fetchall()]
        
        # Get decision points
        cursor.execute("""
            SELECT 
                dp.id, dp.name, dp.description, dp.component,
                pdp.alignment_threshold, pdp.priority
            FROM decision_points dp
            JOIN principle_decision_points pdp ON dp.id = pdp.decision_point_id
            WHERE pdp.principle_id = ?
            ORDER BY pdp.priority DESC, dp.name
        """, (principle_id,))
        principle_dict["decision_points"] = [dict(row) for row in cursor.fetchall()]
        
        # Get dependencies
        cursor.execute("""
            SELECT 
                p2.id, p2.name, p2.short_name,
                pd.relationship_type, pd.strength, pd.description
            FROM principles p2
            JOIN principle_dependencies pd ON p2.id = pd.dependent_principle_id
            WHERE pd.principle_id = ?
            ORDER BY pd.strength DESC, p2.name
        """, (principle_id,))
        principle_dict["supports"] = [dict(row) for row in cursor.fetchall() if row["relationship_type"] == "supports"]
        principle_dict["requires"] = [dict(row) for row in cursor.fetchall() if row["relationship_type"] == "requires"]
        principle_dict["conflicts"] = [dict(row) for row in cursor.fetchall() if row["relationship_type"] == "conflicts"]
        
        conn.close()
        return principle_dict
    
    def create_principle(
        self,
        name: str,
        description: str,
        category_id: int,
        importance_level_id: int,
        short_name: Optional[str] = None,
        evaluation_criteria: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        decision_points: Optional[List[Dict[str, Any]]] = None,
        is_active: bool = True,
        user_id: Optional[str] = None
    ) -> int:
        """
        Create a new principle with optional related data.
        
        Args:
            name: Principle name
            description: Principle description
            category_id: Category ID
            importance_level_id: Importance level ID
            short_name: Optional short name for programmatic reference
            evaluation_criteria: Optional list of evaluation criteria
            tags: Optional list of tag names
            decision_points: Optional list of decision point configurations
            is_active: Whether the principle is active
            user_id: Optional ID of the user creating the principle
            
        Returns:
            ID of the created principle
            
        Raises:
            ValueError: If a principle with the same name already exists
            sqlite3.IntegrityError: If referenced entities don't exist
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if principle with the same name exists
            cursor.execute("SELECT id FROM principles WHERE name = ?", (name,))
            if cursor.fetchone():
                raise ValueError(f"Principle with name '{name}' already exists")
            
            # Check if short_name is provided and unique
            if short_name:
                cursor.execute("SELECT id FROM principles WHERE short_name = ?", (short_name,))
                if cursor.fetchone():
                    raise ValueError(f"Principle with short_name '{short_name}' already exists")
            
            # Insert principle
            cursor.execute(
                """
                INSERT INTO principles (
                    name, description, category_id, importance_level_id, short_name,
                    is_active, created_by, updated_by, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (name, description, category_id, importance_level_id, short_name, 
                 1 if is_active else 0, user_id, user_id)
            )
            principle_id = cursor.lastrowid
            
            # Add evaluation criteria
            if evaluation_criteria:
                for ec in evaluation_criteria:
                    cursor.execute(
                        """
                        INSERT INTO evaluation_criteria (
                            principle_id, type_id, content, parameters,
                            version, is_active, created_by, updated_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            principle_id, 
                            ec["type_id"], 
                            ec["content"], 
                            json.dumps(ec.get("parameters", {})),
                            ec.get("version", "1.0"),
                            1 if ec.get("is_active", True) else 0,
                            user_id, 
                            user_id
                        )
                    )
            
            # Add tags
            if tags:
                for tag_name in tags:
                    # Get or create tag
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
                    tag_row = cursor.fetchone()
                    if tag_row:
                        tag_id = tag_row["id"]
                    else:
                        cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
                        tag_id = cursor.lastrowid
                    
                    # Link tag to principle
                    cursor.execute(
                        "INSERT INTO principle_tags (principle_id, tag_id) VALUES (?, ?)",
                        (principle_id, tag_id)
                    )
            
            # Add decision points
            if decision_points:
                for dp in decision_points:
                    cursor.execute(
                        """
                        INSERT INTO principle_decision_points (
                            principle_id, decision_point_id, alignment_threshold, priority
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            principle_id,
                            dp["decision_point_id"],
                            dp.get("alignment_threshold", 0.7),
                            dp.get("priority", 0)
                        )
                    )
            
            # Record audit
            self._record_audit(
                conn=conn,
                table_name="principles",
                record_id=principle_id,
                action="INSERT",
                new_values={
                    "name": name,
                    "description": description,
                    "category_id": category_id,
                    "importance_level_id": importance_level_id,
                    "short_name": short_name,
                    "is_active": is_active
                },
                user_id=user_id
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Created principle: {name} (ID: {principle_id})")
            return principle_id
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error creating principle: {str(e)}")
            raise
    
    def update_principle(
        self,
        principle_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category_id: Optional[int] = None,
        importance_level_id: Optional[int] = None,
        short_name: Optional[str] = None,
        is_active: Optional[bool] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update a principle.
        
        Args:
            principle_id: ID of the principle to update
            name: Optional new name
            description: Optional new description
            category_id: Optional new category ID
            importance_level_id: Optional new importance level ID
            short_name: Optional new short name
            is_active: Optional new active status
            user_id: Optional ID of the user updating the principle
            
        Returns:
            True if the principle was updated, False if not found
            
        Raises:
            ValueError: If a different principle with the same name already exists
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute(
                """
                SELECT name, description, category_id, importance_level_id, 
                       short_name, version, is_active
                FROM principles 
                WHERE id = ?
                """,
                (principle_id,)
            )
            current = cursor.fetchone()
            if not current:
                conn.close()
                return False
            
            current_dict = dict(current)
            
            # Check if name is being changed and if a different principle with the same name exists
            if name and name != current_dict["name"]:
                cursor.execute(
                    "SELECT id FROM principles WHERE name = ? AND id <> ?",
                    (name, principle_id)
                )
                if cursor.fetchone():
                    raise ValueError(f"Different principle with name '{name}' already exists")
            
            # Check if short_name is being changed and if a different principle with the same short_name exists
            if short_name and short_name != current_dict["short_name"]:
                cursor.execute(
                    "SELECT id FROM principles WHERE short_name = ? AND id <> ?",
                    (short_name, principle_id)
                )
                if cursor.fetchone():
                    raise ValueError(f"Different principle with short_name '{short_name}' already exists")
            
            # Build update parts
            update_parts = []
            params = []
            
            if name is not None:
                update_parts.append("name = ?")
                params.append(name)
            
            if description is not None:
                update_parts.append("description = ?")
                params.append(description)
            
            if category_id is not None:
                update_parts.append("category_id = ?")
                params.append(category_id)
            
            if importance_level_id is not None:
                update_parts.append("importance_level_id = ?")
                params.append(importance_level_id)
            
            if short_name is not None:
                update_parts.append("short_name = ?")
                params.append(short_name)
            
            if is_active is not None:
                update_parts.append("is_active = ?")
                params.append(1 if is_active else 0)
            
            # Only update if there are changes
            if update_parts:
                update_parts.append("updated_at = CURRENT_TIMESTAMP")
                update_parts.append("updated_by = ?")
                params.append(user_id)
                
                query = f"UPDATE principles SET {', '.join(update_parts)} WHERE id = ?"
                params.append(principle_id)
                
                cursor.execute(query, params)
                
                # Record audit
                self._record_audit(
                    conn=conn,
                    table_name="principles",
                    record_id=principle_id,
                    action="UPDATE",
                    old_values=current_dict,
                    new_values={
                        "name": name if name is not None else current_dict["name"],
                        "description": description if description is not None else current_dict["description"],
                        "category_id": category_id if category_id is not None else current_dict["category_id"],
                        "importance_level_id": importance_level_id if importance_level_id is not None else current_dict["importance_level_id"],
                        "short_name": short_name if short_name is not None else current_dict["short_name"],
                        "is_active": is_active if is_active is not None else current_dict["is_active"]
                    },
                    user_id=user_id
                )
                
                conn.commit()
                logger.info(f"Updated principle ID {principle_id}")
            
            conn.close()
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error updating principle: {str(e)}")
            raise
    
    def delete_principle(self, principle_id: int, user_id: Optional[str] = None) -> bool:
        """
        Delete a principle.
        
        Args:
            principle_id: ID of the principle to delete
            user_id: Optional ID of the user deleting the principle
            
        Returns:
            True if the principle was deleted, False if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute(
                """
                SELECT name, description, category_id, importance_level_id, 
                       short_name, version, is_active
                FROM principles 
                WHERE id = ?
                """,
                (principle_id,)
            )
            current = cursor.fetchone()
            if not current:
                conn.close()
                return False
            
            # Record audit before deletion
            self._record_audit(
                conn=conn,
                table_name="principles",
                record_id=principle_id,
                action="DELETE",
                old_values=dict(current),
                user_id=user_id
            )
            
            # Delete the principle - all related records will be deleted due to CASCADE constraints
            cursor.execute("DELETE FROM principles WHERE id = ?", (principle_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted principle ID {principle_id}")
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error deleting principle: {str(e)}")
            raise
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Principle Tag Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def manage_principle_tags(
        self,
        principle_id: int,
        tags_to_add: Optional[List[str]] = None,
        tags_to_remove: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Add or remove tags from a principle.
        
        Args:
            principle_id: ID of the principle
            tags_to_add: Optional list of tag names to add
            tags_to_remove: Optional list of tag names to remove
            user_id: Optional ID of the user managing the tags
            
        Returns:
            True if the operation was successful, False if the principle was not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if principle exists
            cursor.execute("SELECT id FROM principles WHERE id = ?", (principle_id,))
            if not cursor.fetchone():
                conn.close()
                return False
            
            # Add tags
            if tags_to_add:
                for tag_name in tags_to_add:
                    # Get or create tag
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
                    tag_row = cursor.fetchone()
                    if tag_row:
                        tag_id = tag_row["id"]
                    else:
                        cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
                        tag_id = cursor.lastrowid
                    
                    # Link tag to principle if not already linked
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO principle_tags (principle_id, tag_id)
                        VALUES (?, ?)
                        """,
                        (principle_id, tag_id)
                    )
            
            # Remove tags
            if tags_to_remove:
                tag_placeholders = ", ".join(["?"] * len(tags_to_remove))
                cursor.execute(
                    f"""
                    DELETE FROM principle_tags
                    WHERE principle_id = ?
                    AND tag_id IN (
                        SELECT id FROM tags WHERE name IN ({tag_placeholders})
                    )
                    """,
                    [principle_id] + tags_to_remove
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated tags for principle ID {principle_id}")
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error managing principle tags: {str(e)}")
            raise
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Evaluation Criteria Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def add_evaluation_criteria(
        self,
        principle_id: int,
        type_id: int,
        content: str,
        parameters: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
        user_id: Optional[str] = None
    ) -> int:
        """
        Add evaluation criteria to a principle.
        
        Args:
            principle_id: ID of the principle
            type_id: ID of the evaluation type
            content: Evaluation content (LLM prompt, rules, etc.)
            parameters: Optional additional parameters
            is_active: Whether the criteria are active
            user_id: Optional ID of the user adding the criteria
            
        Returns:
            ID of the created evaluation criteria
            
        Raises:
            ValueError: If the principle does not exist
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if principle exists
            cursor.execute("SELECT id FROM principles WHERE id = ?", (principle_id,))
            if not cursor.fetchone():
                raise ValueError(f"Principle with ID {principle_id} does not exist")
            
            # Insert criteria
            cursor.execute(
                """
                INSERT INTO evaluation_criteria (
                    principle_id, type_id, content, parameters,
                    is_active, created_by, updated_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    principle_id,
                    type_id,
                    content,
                    json.dumps(parameters or {}),
                    1 if is_active else 0,
                    user_id,
                    user_id
                )
            )
            criteria_id = cursor.lastrowid
            
            # Record audit
            self._record_audit(
                conn=conn,
                table_name="evaluation_criteria",
                record_id=criteria_id,
                action="INSERT",
                new_values={
                    "principle_id": principle_id,
                    "type_id": type_id,
                    "content": content,
                    "parameters": parameters,
                    "is_active": is_active
                },
                user_id=user_id
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added evaluation criteria to principle ID {principle_id} (criteria ID: {criteria_id})")
            return criteria_id
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error adding evaluation criteria: {str(e)}")
            raise
    
    def update_evaluation_criteria(
        self,
        criteria_id: int,
        content: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update evaluation criteria.
        
        Args:
            criteria_id: ID of the evaluation criteria to update
            content: Optional new content
            parameters: Optional new parameters
            is_active: Optional new active status
            user_id: Optional ID of the user updating the criteria
            
        Returns:
            True if the criteria were updated, False if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute(
                """
                SELECT principle_id, type_id, content, parameters, is_active
                FROM evaluation_criteria
                WHERE id = ?
                """,
                (criteria_id,)
            )
            current = cursor.fetchone()
            if not current:
                conn.close()
                return False
            
            current_dict = dict(current)
            current_dict["parameters"] = json.loads(current_dict["parameters"]) if current_dict["parameters"] else {}
            
            # Build update parts
            update_parts = []
            params = []
            
            if content is not None:
                update_parts.append("content = ?")
                params.append(content)
            
            if parameters is not None:
                update_parts.append("parameters = ?")
                params.append(json.dumps(parameters))
            
            if is_active is not None:
                update_parts.append("is_active = ?")
                params.append(1 if is_active else 0)
            
            # Only update if there are changes
            if update_parts:
                update_parts.append("updated_at = CURRENT_TIMESTAMP")
                update_parts.append("updated_by = ?")
                params.append(user_id)
                
                query = f"UPDATE evaluation_criteria SET {', '.join(update_parts)} WHERE id = ?"
                params.append(criteria_id)
                
                cursor.execute(query, params)
                
                # Record audit
                self._record_audit(
                    conn=conn,
                    table_name="evaluation_criteria",
                    record_id=criteria_id,
                    action="UPDATE",
                    old_values=current_dict,
                    new_values={
                        "content": content if content is not None else current_dict["content"],
                        "parameters": parameters if parameters is not None else current_dict["parameters"],
                        "is_active": is_active if is_active is not None else current_dict["is_active"]
                    },
                    user_id=user_id
                )
                
                conn.commit()
                logger.info(f"Updated evaluation criteria ID {criteria_id}")
            
            conn.close()
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error updating evaluation criteria: {str(e)}")
            raise
    
    def delete_evaluation_criteria(
        self,
        criteria_id: int,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete evaluation criteria.
        
        Args:
            criteria_id: ID of the evaluation criteria to delete
            user_id: Optional ID of the user deleting the criteria
            
        Returns:
            True if the criteria were deleted, False if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current values for audit
            cursor.execute(
                """
                SELECT principle_id, type_id, content, parameters, is_active
                FROM evaluation_criteria
                WHERE id = ?
                """,
                (criteria_id,)
            )
            current = cursor.fetchone()
            if not current:
                conn.close()
                return False
            
            current_dict = dict(current)
            current_dict["parameters"] = json.loads(current_dict["parameters"]) if current_dict["parameters"] else {}
            
            # Record audit before deletion
            self._record_audit(
                conn=conn,
                table_name="evaluation_criteria",
                record_id=criteria_id,
                action="DELETE",
                old_values=current_dict,
                user_id=user_id
            )
            
            # Delete the criteria
            cursor.execute("DELETE FROM evaluation_criteria WHERE id = ?", (criteria_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted evaluation criteria ID {criteria_id}")
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error deleting evaluation criteria: {str(e)}")
            raise
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Decision Point Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def get_decision_points(self) -> List[Dict[str, Any]]:
        """
        Get all decision points.
        
        Returns:
            List of decision point dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, description, component, context_schema, created_at, updated_at "
            "FROM decision_points "
            "ORDER BY name"
        )
        decision_points = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return decision_points
    
    def get_decision_point(self, decision_point_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a decision point by ID.
        
        Args:
            decision_point_id: Decision point ID
            
        Returns:
            Decision point dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, description, component, context_schema, created_at, updated_at "
            "FROM decision_points "
            "WHERE id = ?",
            (decision_point_id,)
        )
        decision_point = cursor.fetchone()
        conn.close()
        
        if decision_point:
            return dict(decision_point)
        return None
    
    def assign_principle_to_decision_point(
        self,
        principle_id: int,
        decision_point_id: int,
        alignment_threshold: float = 0.7,
        priority: int = 0,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Assign a principle to a decision point.
        
        Args:
            principle_id: ID of the principle
            decision_point_id: ID of the decision point
            alignment_threshold: Alignment threshold for principle evaluation
            priority: Priority of the principle at this decision point
            user_id: Optional ID of the user making the assignment
            
        Returns:
            True if the assignment was created or updated
            
        Raises:
            ValueError: If the principle or decision point does not exist
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if principle exists
            cursor.execute("SELECT id FROM principles WHERE id = ?", (principle_id,))
            if not cursor.fetchone():
                raise ValueError(f"Principle with ID {principle_id} does not exist")
            
            # Check if decision point exists
            cursor.execute("SELECT id FROM decision_points WHERE id = ?", (decision_point_id,))
            if not cursor.fetchone():
                raise ValueError(f"Decision point with ID {decision_point_id} does not exist")
            
            # Check if assignment already exists
            cursor.execute(
                """
                SELECT alignment_threshold, priority
                FROM principle_decision_points
                WHERE principle_id = ? AND decision_point_id = ?
                """,
                (principle_id, decision_point_id)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing assignment
                cursor.execute(
                    """
                    UPDATE principle_decision_points
                    SET alignment_threshold = ?, priority = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE principle_id = ? AND decision_point_id = ?
                    """,
                    (alignment_threshold, priority, principle_id, decision_point_id)
                )
                
                # Record audit
                self._record_audit(
                    conn=conn,
                    table_name="principle_decision_points",
                    record_id=f"{principle_id}_{decision_point_id}",
                    action="UPDATE",
                    old_values=dict(existing),
                    new_values={
                        "alignment_threshold": alignment_threshold,
                        "priority": priority
                    },
                    user_id=user_id
                )
                
                logger.info(f"Updated assignment of principle ID {principle_id} to decision point ID {decision_point_id}")
            else:
                # Create new assignment
                cursor.execute(
                    """
                    INSERT INTO principle_decision_points (
                        principle_id, decision_point_id, alignment_threshold, priority
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (principle_id, decision_point_id, alignment_threshold, priority)
                )
                
                # Record audit
                self._record_audit(
                    conn=conn,
                    table_name="principle_decision_points",
                    record_id=f"{principle_id}_{decision_point_id}",
                    action="INSERT",
                    new_values={
                        "principle_id": principle_id,
                        "decision_point_id": decision_point_id,
                        "alignment_threshold": alignment_threshold,
                        "priority": priority
                    },
                    user_id=user_id
                )
                
                logger.info(f"Assigned principle ID {principle_id} to decision point ID {decision_point_id}")
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error assigning principle to decision point: {str(e)}")
            raise
    
    def remove_principle_from_decision_point(
        self,
        principle_id: int,
        decision_point_id: int,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Remove a principle from a decision point.
        
        Args:
            principle_id: ID of the principle
            decision_point_id: ID of the decision point
            user_id: Optional ID of the user making the removal
            
        Returns:
            True if the assignment was removed, False if it didn't exist
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if assignment exists
            cursor.execute(
                """
                SELECT alignment_threshold, priority
                FROM principle_decision_points
                WHERE principle_id = ? AND decision_point_id = ?
                """,
                (principle_id, decision_point_id)
            )
            existing = cursor.fetchone()
            
            if not existing:
                conn.close()
                return False
            
            # Record audit before deletion
            self._record_audit(
                conn=conn,
                table_name="principle_decision_points",
                record_id=f"{principle_id}_{decision_point_id}",
                action="DELETE",
                old_values=dict(existing),
                user_id=user_id
            )
            
            # Remove assignment
            cursor.execute(
                """
                DELETE FROM principle_decision_points
                WHERE principle_id = ? AND decision_point_id = ?
                """,
                (principle_id, decision_point_id)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Removed principle ID {principle_id} from decision point ID {decision_point_id}")
            return True
        except Exception as e:
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            logger.error(f"Error removing principle from decision point: {str(e)}")
            raise
