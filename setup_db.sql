-- Database Initialization Script for Multimodal Emotional Support Chatbot
CREATE DATABASE IF NOT EXISTS emotional_support_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE emotional_support_db;

-- 1. Anonymous Chat Sessions Table
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    current_mood VARCHAR(50) DEFAULT 'neutral'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Multimodal Chat Messages Table
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    sender ENUM('user', 'assistant', 'system') NOT NULL,
    text_content TEXT,
    has_image BOOLEAN DEFAULT FALSE,
    image_path VARCHAR(255) DEFAULT NULL,
    has_audio BOOLEAN DEFAULT FALSE,
    audio_path VARCHAR(255) DEFAULT NULL,
    sentiment_tag VARCHAR(50) DEFAULT 'neutral',
    guardrail_flagged BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    CONSTRAINT fk_session FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Safety and Guardrail Trigger Audit Logs
CREATE TABLE IF NOT EXISTS guardrail_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    trigger_type ENUM('self_harm', 'violence', 'medical_advice_seeking', 'pii_leak', 'toxic_language') NOT NULL,
    severity ENUM('low', 'medium', 'critical') NOT NULL,
    user_input_snippet TEXT,
    mitigation_action VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_guardrail_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
