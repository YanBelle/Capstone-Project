"""
Comprehensive Clear Data Function for ABM Anomaly Detection System
Handles foreign key constraints with multiple fallback strategies
"""

async def clear_all_data_comprehensive(confirm: str = None):
    """
    Clear all data from the system with robust foreign key constraint handling
    """
    if confirm != "true":
        raise HTTPException(
            status_code=400, 
            detail="This operation requires confirmation. Add ?confirm=true to proceed."
        )
    
    import os
    import shutil
    import glob
    
    cleared_summary = {
        "database_tables_cleared": [],
        "files_cleared": [],
        "redis_cleared": False,
        "method_used": "",
        "errors": []
    }
    
    try:
        # METHOD 1: Transaction with explicit ordering
        with db_engine.connect() as conn:
            try:
                trans = conn.begin()
                
                # Clear tables in strict dependency order
                deletion_order = [
                    ("ml_anomalies", "Child table referencing ml_sessions"),
                    ("expert_feedback", "Child table referencing ml_sessions"), 
                    ("labeled_anomalies", "Child table referencing ml_sessions"),
                    ("anomaly_detections", "Independent table"),
                    ("ml_summaries", "Independent table"),
                    ("ml_sessions", "Parent table")
                ]
                
                for table_name, description in deletion_order:
                    try:
                        result = conn.execute(text(f"DELETE FROM {table_name}"))
                        row_count = result.rowcount
                        cleared_summary["database_tables_cleared"].append(f"{table_name} ({row_count} rows)")
                        logger.info(f"Cleared {table_name}: {row_count} rows")
                    except Exception as table_error:
                        error_msg = f"Failed to clear {table_name}: {str(table_error)}"
                        cleared_summary["errors"].append(error_msg)
                        logger.error(error_msg)
                        raise table_error  # Stop on first error for transaction
                
                trans.commit()
                cleared_summary["method_used"] = "Transaction with dependency order"
                logger.info("Method 1 (Transaction) succeeded")
                
            except Exception as method1_error:
                trans.rollback()
                logger.warning(f"Method 1 failed: {method1_error}")
                
                # METHOD 2: TRUNCATE CASCADE
                try:
                    logger.info("Attempting Method 2: TRUNCATE CASCADE")
                    cleared_summary["database_tables_cleared"] = []
                    
                    # TRUNCATE with CASCADE handles foreign keys automatically
                    all_tables = ["ml_sessions", "ml_anomalies", "expert_feedback", 
                                 "labeled_anomalies", "anomaly_detections", "ml_summaries"]
                    
                    for table in all_tables:
                        try:
                            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
                            cleared_summary["database_tables_cleared"].append(f"{table} (truncated)")
                            logger.info(f"Truncated {table}")
                        except Exception as truncate_error:
                            error_msg = f"Could not truncate {table}: {str(truncate_error)}"
                            cleared_summary["errors"].append(error_msg)
                            logger.warning(error_msg)
                    
                    conn.commit()
                    cleared_summary["method_used"] = "TRUNCATE CASCADE"
                    logger.info("Method 2 (TRUNCATE CASCADE) succeeded")
                    
                except Exception as method2_error:
                    logger.warning(f"Method 2 failed: {method2_error}")
                    
                    # METHOD 3: Drop constraints, delete, recreate constraints
                    try:
                        logger.info("Attempting Method 3: Temporary constraint removal")
                        cleared_summary["database_tables_cleared"] = []
                        
                        # Drop foreign key constraint temporarily
                        try:
                            conn.execute(text("ALTER TABLE ml_anomalies DROP CONSTRAINT IF EXISTS ml_anomalies_session_id_fkey"))
                            logger.info("Dropped foreign key constraint")
                        except Exception as drop_error:
                            logger.warning(f"Could not drop constraint: {drop_error}")
                        
                        # Delete all data
                        for table in all_tables:
                            try:
                                result = conn.execute(text(f"DELETE FROM {table}"))
                                row_count = result.rowcount
                                cleared_summary["database_tables_cleared"].append(f"{table} ({row_count} rows)")
                                logger.info(f"Cleared {table}: {row_count} rows")
                            except Exception as delete_error:
                                error_msg = f"Could not clear {table}: {str(delete_error)}"
                                cleared_summary["errors"].append(error_msg)
                                logger.warning(error_msg)
                        
                        # Recreate foreign key constraint
                        try:
                            conn.execute(text("""
                                ALTER TABLE ml_anomalies 
                                ADD CONSTRAINT ml_anomalies_session_id_fkey 
                                FOREIGN KEY (session_id) REFERENCES ml_sessions(session_id)
                            """))
                            logger.info("Recreated foreign key constraint")
                        except Exception as recreate_error:
                            logger.warning(f"Could not recreate constraint: {recreate_error}")
                        
                        conn.commit()
                        cleared_summary["method_used"] = "Temporary constraint removal"
                        logger.info("Method 3 (Constraint removal) succeeded")
                        
                    except Exception as method3_error:
                        logger.error(f"All database clearing methods failed: {method3_error}")
                        cleared_summary["errors"].append(f"Database clearing failed: {str(method3_error)}")
        
        # Clear file system data (regardless of database clearing success)
        try:
            file_dirs = [
                ("/app/data/sessions", "sessions"),
                ("/app/data/output", "output"),
                ("/app/data/processed", "processed"),
                ("/app/data/models", "models"),
                ("/app/data/logs", "logs"),
                ("/app/static/debug", "debug"),
                ("/app/uploads", "uploads")
            ]
            
            for dir_path, dir_name in file_dirs:
                try:
                    if os.path.exists(dir_path):
                        files = glob.glob(os.path.join(dir_path, "*"))
                        if files:
                            file_count = len(files)
                            shutil.rmtree(dir_path)
                            os.makedirs(dir_path, exist_ok=True)
                            cleared_summary["files_cleared"].append(f"{dir_name} ({file_count} files)")
                            logger.info(f"Cleared {dir_path}: {file_count} files")
                except Exception as file_error:
                    error_msg = f"Could not clear {dir_path}: {str(file_error)}"
                    cleared_summary["errors"].append(error_msg)
                    logger.warning(error_msg)
            
        except Exception as file_system_error:
            error_msg = f"File system clearing error: {str(file_system_error)}"
            cleared_summary["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Clear Redis cache
        try:
            import redis
            redis_hosts = ['redis', 'localhost', '127.0.0.1']
            
            for host in redis_hosts:
                try:
                    r = redis.Redis(host=host, port=6379, db=0, socket_timeout=5)
                    r.ping()
                    r.flushall()
                    cleared_summary["redis_cleared"] = True
                    logger.info(f"Redis cache cleared (host: {host})")
                    break
                except Exception as redis_host_error:
                    logger.debug(f"Redis connection failed for {host}: {redis_host_error}")
                    continue
            
            if not cleared_summary["redis_cleared"]:
                cleared_summary["errors"].append("Could not connect to Redis")
                
        except Exception as redis_error:
            error_msg = f"Redis clearing error: {str(redis_error)}"
            cleared_summary["errors"].append(error_msg)
            logger.warning(error_msg)
        
        # Prepare response
        success = len(cleared_summary["database_tables_cleared"]) > 0 or len(cleared_summary["files_cleared"]) > 0
        
        return {
            "status": "success" if success else "partial",
            "message": "Data clearing completed" + (" with some errors" if cleared_summary["errors"] else " successfully"),
            "database_tables_cleared": cleared_summary["database_tables_cleared"],
            "files_cleared": cleared_summary["files_cleared"],
            "redis_cleared": cleared_summary["redis_cleared"],
            "method_used": cleared_summary["method_used"],
            "total_tables": len(cleared_summary["database_tables_cleared"]),
            "total_file_groups": len(cleared_summary["files_cleared"]),
            "errors": cleared_summary["errors"] if cleared_summary["errors"] else None
        }
        
    except Exception as e:
        logger.error(f"Clear data operation failed completely: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear data failed: {str(e)}")
