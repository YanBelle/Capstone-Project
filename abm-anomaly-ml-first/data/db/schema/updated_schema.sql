--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13
-- Dumped by pg_dump version 15.13 (Ubuntu 15.13-1.pgdg20.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alerts; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.alerts (
    id integer NOT NULL,
    anomaly_id integer,
    alert_level character varying(20),
    message text,
    is_resolved boolean DEFAULT false,
    resolved_at timestamp without time zone,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.alerts OWNER TO abm_user;

--
-- Name: alerts_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.alerts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alerts_id_seq OWNER TO abm_user;

--
-- Name: alerts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.alerts_id_seq OWNED BY public.alerts.id;


--
-- Name: anomaly_clusters; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.anomaly_clusters (
    id integer NOT NULL,
    cluster_id integer NOT NULL,
    cluster_name character varying(100),
    cluster_description text,
    typical_patterns jsonb,
    member_count integer DEFAULT 0,
    centroid_vector bytea,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.anomaly_clusters OWNER TO abm_user;

--
-- Name: anomaly_clusters_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.anomaly_clusters_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.anomaly_clusters_id_seq OWNER TO abm_user;

--
-- Name: anomaly_clusters_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.anomaly_clusters_id_seq OWNED BY public.anomaly_clusters.id;


--
-- Name: anomaly_patterns; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.anomaly_patterns (
    id integer NOT NULL,
    pattern_name character varying(100) NOT NULL,
    pattern_regex text,
    pattern_description text,
    severity_level character varying(20),
    recommended_action text,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.anomaly_patterns OWNER TO abm_user;

--
-- Name: anomaly_patterns_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.anomaly_patterns_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.anomaly_patterns_id_seq OWNER TO abm_user;

--
-- Name: anomaly_patterns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.anomaly_patterns_id_seq OWNED BY public.anomaly_patterns.id;


--
-- Name: expert_feedback; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.expert_feedback (
    id integer NOT NULL,
    session_id character varying(255) NOT NULL,
    expert_label character varying(100) NOT NULL,
    expert_confidence double precision NOT NULL,
    feedback_type character varying(50) NOT NULL,
    expert_explanation text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100) DEFAULT 'expert_user'::character varying
);


ALTER TABLE public.expert_feedback OWNER TO abm_user;

--
-- Name: expert_feedback_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.expert_feedback_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.expert_feedback_id_seq OWNER TO abm_user;

--
-- Name: expert_feedback_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.expert_feedback_id_seq OWNED BY public.expert_feedback.id;


--
-- Name: flyway_schema_history; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.flyway_schema_history (
    installed_rank integer NOT NULL,
    version character varying(50),
    description character varying(200) NOT NULL,
    type character varying(20) NOT NULL,
    script character varying(1000) NOT NULL,
    checksum integer,
    installed_by character varying(100) NOT NULL,
    installed_on timestamp without time zone DEFAULT now() NOT NULL,
    execution_time integer NOT NULL,
    success boolean NOT NULL
);


ALTER TABLE public.flyway_schema_history OWNER TO abm_user;

--
-- Name: labeled_anomalies; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.labeled_anomalies (
    id integer NOT NULL,
    session_id character varying(100),
    anomaly_label character varying(100) NOT NULL,
    label_confidence numeric(3,2),
    labeled_by character varying(100),
    label_reason text,
    is_verified boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.labeled_anomalies OWNER TO abm_user;

--
-- Name: labeled_anomalies_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.labeled_anomalies_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.labeled_anomalies_id_seq OWNER TO abm_user;

--
-- Name: labeled_anomalies_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.labeled_anomalies_id_seq OWNED BY public.labeled_anomalies.id;


--
-- Name: ml_anomalies; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.ml_anomalies (
    id integer NOT NULL,
    session_id character varying(100),
    anomaly_type character varying(100),
    anomaly_score numeric(5,4),
    cluster_id integer,
    detected_patterns jsonb,
    critical_events jsonb,
    error_codes jsonb,
    model_name character varying(100),
    model_version character varying(20),
    detected_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.ml_anomalies OWNER TO abm_user;

--
-- Name: ml_anomalies_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.ml_anomalies_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ml_anomalies_id_seq OWNER TO abm_user;

--
-- Name: ml_anomalies_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.ml_anomalies_id_seq OWNED BY public.ml_anomalies.id;


--
-- Name: ml_sessions; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.ml_sessions (
    id integer NOT NULL,
    session_id character varying(100) NOT NULL,
    "timestamp" timestamp without time zone,
    session_length integer,
    is_anomaly boolean DEFAULT false,
    anomaly_score numeric(5,4),
    anomaly_type character varying(100),
    detected_patterns jsonb,
    critical_events jsonb,
    embedding_vector bytea,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.ml_sessions OWNER TO abm_user;

--
-- Name: ml_anomaly_summary; Type: VIEW; Schema: public; Owner: abm_user
--

CREATE VIEW public.ml_anomaly_summary AS
 SELECT date(s."timestamp") AS date,
    count(DISTINCT s.id) AS total_sessions,
    count(DISTINCT
        CASE
            WHEN s.is_anomaly THEN s.id
            ELSE NULL::integer
        END) AS anomaly_count,
    avg(
        CASE
            WHEN s.is_anomaly THEN s.anomaly_score
            ELSE NULL::numeric
        END) AS avg_anomaly_score,
    count(DISTINCT a.cluster_id) AS unique_clusters,
    array_agg(DISTINCT a.anomaly_type) AS anomaly_types
   FROM (public.ml_sessions s
     LEFT JOIN public.ml_anomalies a ON (((s.session_id)::text = (a.session_id)::text)))
  GROUP BY (date(s."timestamp"));


ALTER TABLE public.ml_anomaly_summary OWNER TO abm_user;

--
-- Name: ml_models; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.ml_models (
    id integer NOT NULL,
    model_name character varying(100) NOT NULL,
    model_type character varying(50),
    model_version character varying(20),
    training_date timestamp without time zone,
    training_samples integer,
    anomaly_threshold numeric(5,4),
    performance_metrics jsonb,
    model_parameters jsonb,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.ml_models OWNER TO abm_user;

--
-- Name: ml_models_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.ml_models_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ml_models_id_seq OWNER TO abm_user;

--
-- Name: ml_models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.ml_models_id_seq OWNED BY public.ml_models.id;


--
-- Name: ml_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.ml_sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ml_sessions_id_seq OWNER TO abm_user;

--
-- Name: ml_sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.ml_sessions_id_seq OWNED BY public.ml_sessions.id;


--
-- Name: model_retraining_events; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.model_retraining_events (
    id integer NOT NULL,
    trigger_type character varying(50) NOT NULL,
    feedback_samples integer,
    trigger_time timestamp without time zone NOT NULL,
    completion_time timestamp without time zone,
    status character varying(50) NOT NULL,
    performance_improvement double precision,
    error_message text
);


ALTER TABLE public.model_retraining_events OWNER TO abm_user;

--
-- Name: model_retraining_events_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.model_retraining_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.model_retraining_events_id_seq OWNER TO abm_user;

--
-- Name: model_retraining_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.model_retraining_events_id_seq OWNED BY public.model_retraining_events.id;


--
-- Name: transactions; Type: TABLE; Schema: public; Owner: abm_user
--

CREATE TABLE public.transactions (
    id integer NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    card_number character varying(20),
    transaction_type character varying(50),
    amount numeric(10,2),
    status character varying(20),
    error_type character varying(50),
    response_time integer,
    terminal_id character varying(20),
    session_id character varying(50),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.transactions OWNER TO abm_user;

--
-- Name: transactions_id_seq; Type: SEQUENCE; Schema: public; Owner: abm_user
--

CREATE SEQUENCE public.transactions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.transactions_id_seq OWNER TO abm_user;

--
-- Name: transactions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: abm_user
--

ALTER SEQUENCE public.transactions_id_seq OWNED BY public.transactions.id;


--
-- Name: alerts id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.alerts ALTER COLUMN id SET DEFAULT nextval('public.alerts_id_seq'::regclass);


--
-- Name: anomaly_clusters id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.anomaly_clusters ALTER COLUMN id SET DEFAULT nextval('public.anomaly_clusters_id_seq'::regclass);


--
-- Name: anomaly_patterns id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.anomaly_patterns ALTER COLUMN id SET DEFAULT nextval('public.anomaly_patterns_id_seq'::regclass);


--
-- Name: expert_feedback id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.expert_feedback ALTER COLUMN id SET DEFAULT nextval('public.expert_feedback_id_seq'::regclass);


--
-- Name: labeled_anomalies id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.labeled_anomalies ALTER COLUMN id SET DEFAULT nextval('public.labeled_anomalies_id_seq'::regclass);


--
-- Name: ml_anomalies id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_anomalies ALTER COLUMN id SET DEFAULT nextval('public.ml_anomalies_id_seq'::regclass);


--
-- Name: ml_models id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_models ALTER COLUMN id SET DEFAULT nextval('public.ml_models_id_seq'::regclass);


--
-- Name: ml_sessions id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_sessions ALTER COLUMN id SET DEFAULT nextval('public.ml_sessions_id_seq'::regclass);


--
-- Name: model_retraining_events id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.model_retraining_events ALTER COLUMN id SET DEFAULT nextval('public.model_retraining_events_id_seq'::regclass);


--
-- Name: transactions id; Type: DEFAULT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.transactions ALTER COLUMN id SET DEFAULT nextval('public.transactions_id_seq'::regclass);


--
-- Name: alerts alerts_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.alerts
    ADD CONSTRAINT alerts_pkey PRIMARY KEY (id);


--
-- Name: anomaly_clusters anomaly_clusters_cluster_id_key; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.anomaly_clusters
    ADD CONSTRAINT anomaly_clusters_cluster_id_key UNIQUE (cluster_id);


--
-- Name: anomaly_clusters anomaly_clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.anomaly_clusters
    ADD CONSTRAINT anomaly_clusters_pkey PRIMARY KEY (id);


--
-- Name: anomaly_patterns anomaly_patterns_pattern_name_key; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.anomaly_patterns
    ADD CONSTRAINT anomaly_patterns_pattern_name_key UNIQUE (pattern_name);


--
-- Name: anomaly_patterns anomaly_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.anomaly_patterns
    ADD CONSTRAINT anomaly_patterns_pkey PRIMARY KEY (id);


--
-- Name: expert_feedback expert_feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.expert_feedback
    ADD CONSTRAINT expert_feedback_pkey PRIMARY KEY (id);


--
-- Name: flyway_schema_history flyway_schema_history_pk; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.flyway_schema_history
    ADD CONSTRAINT flyway_schema_history_pk PRIMARY KEY (installed_rank);


--
-- Name: labeled_anomalies labeled_anomalies_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.labeled_anomalies
    ADD CONSTRAINT labeled_anomalies_pkey PRIMARY KEY (id);


--
-- Name: ml_anomalies ml_anomalies_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_anomalies
    ADD CONSTRAINT ml_anomalies_pkey PRIMARY KEY (id);


--
-- Name: ml_models ml_models_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_models
    ADD CONSTRAINT ml_models_pkey PRIMARY KEY (id);


--
-- Name: ml_sessions ml_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_sessions
    ADD CONSTRAINT ml_sessions_pkey PRIMARY KEY (id);


--
-- Name: ml_sessions ml_sessions_session_id_key; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_sessions
    ADD CONSTRAINT ml_sessions_session_id_key UNIQUE (session_id);


--
-- Name: model_retraining_events model_retraining_events_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.model_retraining_events
    ADD CONSTRAINT model_retraining_events_pkey PRIMARY KEY (id);


--
-- Name: transactions transactions_pkey; Type: CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.transactions
    ADD CONSTRAINT transactions_pkey PRIMARY KEY (id);


--
-- Name: flyway_schema_history_s_idx; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX flyway_schema_history_s_idx ON public.flyway_schema_history USING btree (success);


--
-- Name: idx_alerts_level; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_alerts_level ON public.alerts USING btree (alert_level);


--
-- Name: idx_alerts_resolved; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_alerts_resolved ON public.alerts USING btree (is_resolved);


--
-- Name: idx_labeled_anomalies_label; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_labeled_anomalies_label ON public.labeled_anomalies USING btree (anomaly_label);


--
-- Name: idx_ml_sessions_anomaly; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_ml_sessions_anomaly ON public.ml_sessions USING btree (is_anomaly);


--
-- Name: idx_ml_sessions_score; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_ml_sessions_score ON public.ml_sessions USING btree (anomaly_score);


--
-- Name: idx_ml_sessions_type; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_ml_sessions_type ON public.ml_sessions USING btree (anomaly_type);


--
-- Name: idx_transactions_session; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_transactions_session ON public.transactions USING btree (session_id);


--
-- Name: idx_transactions_timestamp; Type: INDEX; Schema: public; Owner: abm_user
--

CREATE INDEX idx_transactions_timestamp ON public.transactions USING btree ("timestamp");


--
-- Name: labeled_anomalies labeled_anomalies_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.labeled_anomalies
    ADD CONSTRAINT labeled_anomalies_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.ml_sessions(session_id);


--
-- Name: ml_anomalies ml_anomalies_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: abm_user
--

ALTER TABLE ONLY public.ml_anomalies
    ADD CONSTRAINT ml_anomalies_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.ml_sessions(session_id);


--
-- PostgreSQL database dump complete
--

