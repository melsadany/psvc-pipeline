################################################################################
# Audio Preprocessing
# Crop audio into task segments based on exact timings from metadata
################################################################################

preprocess_audio <- function(audio_file, participant_id, config, 
                             # skip_acoustics = TRUE, 
                             output_dir) {
  
  log_info("Reading audio file...")
  
  # Determine file type and read
  if (grepl("\\.mp3$", audio_file, ignore.case = TRUE)) {
    audio <- readMP3(audio_file)
  } else if (grepl("\\.wav$", audio_file, ignore.case = TRUE)) {
    audio <- readWave(audio_file)
  } else {
    stop("Unsupported audio format. Use .mp3 or .wav")
  }
  
  log_info("Audio file loaded: {round(length(audio@left)/audio@samp.rate, 2)} seconds")
  
  # Create participant directory
  participant_dir <- file.path(output_dir, participant_id)
  dir.create(participant_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Crop segments based on exact timings from config
  log_info("Cropping audio segments from fixed timings...")
  segments <- crop_audio_segments(audio, config, participant_dir, participant_id)
  
  log_info("  ✓ Cropped {length(segments)} segments")
  
  # # Extract acoustic features if requested (currently not implemented)
  # acoustic_features <- NULL
  # if (!skip_acoustics) {
  #   log_info("Extracting acoustic features...")
  #   acoustic_features <- extract_acoustic_features(segments, participant_id)
  # } else {
  #   log_info("Skipping acoustic features...")
  # }

  return(list(
    segments = segments,
    # acoustics = acoustic_features,
    audio_duration = length(audio@left) / audio@samp.rate
  ))
}

################################################################################
# Crop Audio Segments (using exact timings - no onset detection needed)
################################################################################

crop_audio_segments <- function(audio, config, output_dir, participant_id) {
  
  segments <- list()
  
  ################################################################################
  # Task 1: WAT (Semantic Fluency)
  ################################################################################
  if (!is.null(config$tasks$WAT)) {
    task_info <- config$tasks$WAT
    
    for (i in seq_along(task_info$prompts)) {
      prompt <- task_info$prompts[[i]]
      
      segment <- extractWave(audio, 
                             from = prompt$start_sec, 
                             to = prompt$end_sec,
                             xunit = "time")
      
      # Save segment
      segment_file <- file.path(output_dir, 
                                paste0(participant_id, 
                                       "_task-", task_info$task_num,
                                       "_", prompt$word, 
                                       "_", i, ".wav"))
      
      writeWave(Wave(left = as.numeric(segment@left),
                     samp.rate = audio@samp.rate,
                     bit = 16,
                     pcm = TRUE),
                filename = segment_file,
                extensible = TRUE)
      
      segments[[length(segments) + 1]] <- list(
        file = segment_file,
        prompt = prompt$word,
        task_type = "WAT",
        task_num = task_info$task_num,
        start_time = prompt$start_sec,
        end_time = prompt$end_sec,
        duration = prompt$end_sec - prompt$start_sec
      )
    }
    
    log_debug("  ✓ Semantic Fluency: {length(task_info$prompts)} word prompts")
  }
  
  ################################################################################
  # Task 2: RAN (Rapid Automatic Naming)
  ################################################################################
  if (!is.null(config$tasks$RAN)) {
    task_info <- config$tasks$RAN
    
    for (i in seq_along(task_info$prompts)) {
      prompt <- task_info$prompts[[i]]
      
      segment <- extractWave(audio, 
                             from = prompt$start_sec, 
                             to = prompt$end_sec,
                             xunit = "time")
      
      # Save segment
      segment_file <- file.path(output_dir, 
                                paste0(participant_id, 
                                       "_task-", task_info$task_num,
                                       "_", prompt$type, 
                                       "_block", prompt$block, ".wav"))
      
      writeWave(Wave(left = as.numeric(segment@left),
                     samp.rate = audio@samp.rate,
                     bit = 16,
                     pcm = TRUE),
                filename = segment_file,
                extensible = TRUE)
      
      segments[[length(segments) + 1]] <- list(
        file = segment_file,
        prompt = prompt$type,
        task_type = "RAN",
        task_num = task_info$task_num,
        block = prompt$block,
        start_time = prompt$start_sec,
        end_time = prompt$end_sec,
        duration = prompt$end_sec - prompt$start_sec
      )
    }
    
    log_debug("  ✓ RAN: {length(task_info$prompts)} blocks")
  }
  
  ################################################################################
  # Task 3: COWAT (Phonemic Fluency)
  ################################################################################
  if (!is.null(config$tasks$COWAT)) {
    task_info <- config$tasks$COWAT
    
    for (i in seq_along(task_info$prompts)) {
      prompt <- task_info$prompts[[i]]
      
      segment <- extractWave(audio, 
                             from = prompt$start_sec, 
                             to = prompt$end_sec,
                             xunit = "time")
      
      # Save segment
      segment_file <- file.path(output_dir, 
                                paste0(participant_id, 
                                       "_task-", task_info$task_num,
                                       "_", prompt$letter,
                                       "_", i, ".wav"))
      
      writeWave(Wave(left = as.numeric(segment@left),
                     samp.rate = audio@samp.rate,
                     bit = 16,
                     pcm = TRUE),
                filename = segment_file,
                extensible = TRUE)
      
      segments[[length(segments) + 1]] <- list(
        file = segment_file,
        prompt = prompt$letter,
        task_type = "COWAT",
        task_num = task_info$task_num,
        start_time = prompt$start_sec,
        end_time = prompt$end_sec,
        duration = prompt$end_sec - prompt$start_sec
      )
    }
    
    log_debug("  ✓ Phonemic Fluency: {length(task_info$prompts)} letter prompts")
  }
  
  ################################################################################
  # Task 4: Faces (no transcription, just timing)
  ################################################################################
  if (!is.null(config$tasks$Faces)) {
    task_info <- config$tasks$Faces
    
    for (i in seq_along(task_info$prompts)) {
      prompt <- task_info$prompts[[i]]
      
      # For faces, we just record timing, no audio cropping needed
      segments[[length(segments) + 1]] <- list(
        file = NULL,  # No audio file for faces
        prompt = paste0("faces", prompt$block),
        task_type = "Faces",
        task_num = task_info$task_num,
        block = prompt$block,
        start_time = prompt$start_sec,
        end_time = prompt$end_sec,
        duration = prompt$end_sec - prompt$start_sec
      )
    }
    
    log_debug("  ✓ Faces: {length(task_info$prompts)} blocks (timing only)")
  }
  
  return(segments)
}

################################################################################
# Extract Acoustic Features
################################################################################

extract_acoustic_features <- function(segments, participant_id) {
  
  log_debug("  Extracting acoustic features from segments...")
  
  # Only extract from segments with audio files (WAT, RAN, COWAT)
  audio_segments <- segments[!sapply(segments, function(x) is.null(x$file))]
  
  if (length(audio_segments) == 0) {
    log_warn("  No audio segments to extract acoustic features from")
    return(NULL)
  }
  
  # Placeholder for Surfboard integration
  # In production, this would call Surfboard on each segment
  
  
  acoustic_features <- data.frame(
    participant_id = participant_id,
    # Surfboard features would go here
    # mfcc_mean, formants, pitch_mean, etc.
    stringsAsFactors = FALSE
  )
  
  log_debug("  ✓ Acoustic features extracted")
  
  return(acoustic_features)
}

