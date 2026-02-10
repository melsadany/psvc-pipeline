#!/usr/bin/env Rscript

suppressWarnings(suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(logger)
  library(tidyverse)
}))

get_script_path <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  needle <- "--file="
  match <- grep(needle, cmdArgs)
  if (length(match) > 0) {
    return(dirname(normalizePath(sub(needle, "", cmdArgs[match]))))
  } else {
    return(getwd())
  }
}

script_dir <- get_script_path()

# Parse arguments
option_list <- list(
  make_option(c("--id"), type = "character", default = NULL),
  make_option(c("--transcription_file"), type = "character", default = NULL),
  make_option(c("--config"), type = "character", default = NULL),
  make_option(c("--reference"), type = "character", default = NULL),
  make_option(c("--output"), type = "character", default = NULL)
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Setup
log_threshold(INFO)
log_appender(appender_console)

# Source dependencies
source(file.path(script_dir, "00_initialize.R"))
source(file.path(script_dir, "04_feature_extraction.R"))

# Load and initialize
config <- yaml::read_yaml(opt$config)
pipeline_env <- initialize_pipeline(
  config = config,
  reference_dir = opt$reference,
  mode = "auto",
  where = dirname(script_dir)
)

log_info("STAGE 4: Feature Extraction")
log_info(strrep("-", 80))

# Read cleaned transcription
clean_tx <- read_tsv(opt$transcription_file, show_col_types = FALSE)

# Read transcription cleaning stats
clean_tx_stats <- read_rds(file.path(opt$output, paste0(opt$id, "_transcription_cleaning_stats.rds")))


# Extract features
features <- extract_all_features(
  clean_transcription = clean_tx,
  participant_id = opt$id,
  embeddings = pipeline_env$embeddings,
  archetype_refs = pipeline_env$archetype_refs,
  reference_data = pipeline_env$reference_data,
  config = config,
  rule_violations = clean_tx_stats$rule_violations
)

# Save results
dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)
write_csv(features$per_prompt, file.path(opt$output, paste0(opt$id, "_per_prompt.csv")))
write_csv(features$per_task, file.path(opt$output, paste0(opt$id, "_per_task.csv")))
write_csv(features$per_participant, file.path(opt$output, paste0(opt$id, "_per_participant.csv")))

log_info("  ✓ Features extracted")
log_info("✓ Stage 4 Complete")
