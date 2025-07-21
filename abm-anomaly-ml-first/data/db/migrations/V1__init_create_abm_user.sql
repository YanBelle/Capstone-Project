-- Create the user if it does not exist
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'abm_user') THEN
      CREATE ROLE abm_user WITH LOGIN PASSWORD 'secure_ml_password123';
   END IF;
END
$$;

-- Create the database if needed (optional, usually handled by docker env vars)
-- CREATE DATABASE abm_ml_db OWNER abm_user;

-- Grant permissions if the DB exists already
GRANT ALL PRIVILEGES ON DATABASE abm_ml_db TO abm_user;

-- Grant permissions if needed (adjust schema/table as necessary)
-- GRANT ALL PRIVILEGES ON DATABASE your_database TO abm_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO abm_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO abm_user;