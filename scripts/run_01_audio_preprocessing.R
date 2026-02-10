#!/usr/bin/env Rscript

suppressWarnings(suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(logger)
  library(tidyverse)
  library(tuneR)
}))

# Get script directory
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

# Parse command line arguments
option_list <- list(
  make_option(c("--audio"), type = "character", default = NULL,
              help = "Path to audio file", metavar = "FILE"),
  make_option(c("--id"), type = "character", default = NULL,
              help = "Participant ID", metavar = "ID"),
  make_option(c("--config"), type = "character", default = NULL,
              help = "Path to task configuration YAML"),
  make_option(c("--output"), type = "character", default = NULL,
              help = "Output directory"),
  make_option(c("--verbose"), action = "store_true", default = FALSE,
              help = "Verbose logging")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Validate arguments
if (is.null(opt$audio) || is.null(opt$id) || is.null(opt$config) || is.null(opt$output)) {
  print_help(opt_parser)
  stop("All arguments are required: --audio, --id, --config, --output", call. = FALSE)
}

# Setup logging
log_threshold(if (opt$verbose) DEBUG else INFO)
log_appender(appender_console)

# Source the function file
source(file.path(script_dir, "01_audio_preprocessing.R"))

# Load configuration
config <- yaml::read_yaml(opt$config)

# Execute preprocessing
log_info("STAGE 1: Audio Preprocessing")
log_info(strrep("-", 80))

audio_results <- preprocess_audio(
  audio_file = opt$audio,
  participant_id = opt$id,
  config = config,
  output_dir = opt$output
)

log_info("  ✓ Audio cropped to {length(audio_results$segments)} segments")

# Save segment metadata for downstream steps
saveRDS(
  audio_results,
  file.path(opt$output, opt$id, paste0(opt$id, "_audio_segments.rds"))
)

log_info("✓ Stage 1 Complete")
