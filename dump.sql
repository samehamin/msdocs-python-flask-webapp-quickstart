-- MySQL dump 10.13  Distrib 8.0.30, for Linux (x86_64)
--
-- Host: localhost    Database: nlpmvpdb
-- ------------------------------------------------------
-- Server version	8.0.30-0ubuntu0.22.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `configs`
--

DROP TABLE IF EXISTS `configs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `configs` (
  `config_id` int NOT NULL AUTO_INCREMENT,
  `config_key` varchar(1000) NOT NULL,
  `config_value` varchar(1000) NOT NULL,
  PRIMARY KEY (`config_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `configs`
--

LOCK TABLES `configs` WRITE;
/*!40000 ALTER TABLE `configs` DISABLE KEYS */;
INSERT INTO `configs` VALUES (1,'threshold-UDML','0.85'),(2,'threshold-DSL','0.93'),(3,'threshold-smallTalk','0.8'),(4,'threshold','0.9');
/*!40000 ALTER TABLE `configs` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `conversations`
--

DROP TABLE IF EXISTS `conversations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `conversations` (
  `conv_id` int NOT NULL AUTO_INCREMENT,
  `conv_start` datetime DEFAULT NULL,
  `conv_end` datetime DEFAULT NULL,
  `user_id` int DEFAULT NULL,
  PRIMARY KEY (`conv_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `conversations`
--

LOCK TABLES `conversations` WRITE;
/*!40000 ALTER TABLE `conversations` DISABLE KEYS */;
/*!40000 ALTER TABLE `conversations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `conversations_messages`
--

DROP TABLE IF EXISTS `conversations_messages`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `conversations_messages` (
  `msg_id` int NOT NULL AUTO_INCREMENT,
  `conv_id` int DEFAULT NULL,
  `msg_date` datetime DEFAULT NULL,
  `user_message` text,
  `detected_intent` varchar(5000) DEFAULT NULL,
  `detected_score` int DEFAULT NULL,
  `current_threshold` int DEFAULT NULL,
  PRIMARY KEY (`msg_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `conversations_messages`
--

LOCK TABLES `conversations_messages` WRITE;
/*!40000 ALTER TABLE `conversations_messages` DISABLE KEYS */;
/*!40000 ALTER TABLE `conversations_messages` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `intents`
--

DROP TABLE IF EXISTS `intents`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `intents` (
  `label` int NOT NULL,
  `intent` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `intents`
--

LOCK TABLES `intents` WRITE;
/*!40000 ALTER TABLE `intents` DISABLE KEYS */;
INSERT INTO `intents` VALUES (80,'balance-check'),(81,'add_beneficiary_account'),(82,'create_complaint');
/*!40000 ALTER TABLE `intents` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `intents_small_talk`
--

DROP TABLE IF EXISTS `intents_small_talk`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `intents_small_talk` (
  `label` int NOT NULL,
  `small_talk_intent` varchar(1000) DEFAULT NULL,
  PRIMARY KEY (`label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `intents_small_talk`
--

LOCK TABLES `intents_small_talk` WRITE;
/*!40000 ALTER TABLE `intents_small_talk` DISABLE KEYS */;
INSERT INTO `intents_small_talk` VALUES (1,'greeting'),(2,'general inquiry');
/*!40000 ALTER TABLE `intents_small_talk` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user_utterances`
--

DROP TABLE IF EXISTS `user_utterances`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_utterances` (
  `id` int NOT NULL AUTO_INCREMENT,
  `utterance` text NOT NULL,
  `label` int NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=36 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_utterances`
--

LOCK TABLES `user_utterances` WRITE;
/*!40000 ALTER TABLE `user_utterances` DISABLE KEYS */;
INSERT INTO `user_utterances` VALUES (1,'could you please let me know what is my current account balance?',80),(2,'what is my current account balance?',80),(3,'what is my bank account balance?',80),(4,'how much i have in my account?',80),(5,'كم رصيد حسابي؟',80),(6,'لو سمحت عايز اعرف رصيد حسابي كام؟',80),(7,'ايش رصيدي بالحساب اليوم؟',80),(8,'معايا كام في الحساب؟',80),(9,'can i add new beneficiary to my list?',81),(10,'please add new beneficiary account',81),(11,'I would like to create new beneficiary account',81),(13,'I want to add my brother as an beneficiary person to my app',81),(27,'Can I talk to a customer care agent?',82),(28,'I have a complaint',82),(29,'I want to talk to customer care',82),(30,'لو سمحت عايز اكلم خدمة العملاء',82),(31,'ابغي اتحدث مع موظف اسعاد المتعاملين لو سمحت',82),(32,'لو سمحت عايز اعمل شكوي',82),(33,'I have problem with my account',82),(34,'عندي مشكلة في حسابي ',82),(35,'I have issue can i talk to an agent?',82);
/*!40000 ALTER TABLE `user_utterances` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-09-08  2:00:24
