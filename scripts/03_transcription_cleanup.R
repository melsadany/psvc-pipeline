################################################################################
# Transcription Cleanup
# Three modes: auto, review, strict
# CRITICAL: Rule violations are SAVED AS FEATURES, not just removed
################################################################################

cleanup_transcription <- function(transcription, participant_id, mode, config, output_dir) {
  log_info("Cleaning transcription (mode: {mode})...")
  
  # Initialize cleanup stats
  removed_stats <- list(
    ums = 0,
    repetitions = 0,
    low_confidence = 0,
    comments = 0,
    total_removed = 0
  )
  
  # Prepare transcription
  tx <- transcription %>%
    mutate(
      word = tolower(word),
      prompt = tolower(prompt),
      # Convert numbers to words if needed
      word_clean = ifelse(grepl("[0-9]", word),
                          as.character(english::english(as.numeric(gsub("[^0-9]", "", word)))),
                          word)
    )
  
  ################################################################################
  # MODE-SPECIFIC FILTERING
  ################################################################################
  
  if (mode == "strict") {
    # STRICT MODE: Only keep high-confidence transcriptions
    threshold <- config$transcription$confidence_threshold$strict
    low_conf_rows <- sum(tx$confidence < threshold, na.rm = TRUE)
    tx <- tx %>% dply::filter(confidence >= threshold | is.na(confidence))
    removed_stats$low_confidence <- low_conf_rows
    log_debug("  Strict mode: Removed {low_conf_rows} low-confidence words (threshold: {threshold})")
    
  } else if (mode == "review") {
    # REVIEW MODE: Flag low-confidence words for manual review
    threshold <- config$transcription$confidence_threshold$review
    flagged <- tx %>%
      dply::filter(confidence < threshold | is.na(confidence)) %>%
      mutate(review_reason = "low_confidence")
    
    if (nrow(flagged) > 0) {
      # Create review file
      review_file <- file.path(output_dir, paste0(participant_id, "_REVIEW_REQUIRED.xlsx"))
      review_df <- flagged %>%
        select(participant_id, prompt, task_number, start, end,
               word, confidence, review_reason) %>%
        mutate(
          corrected_word = "",  # Empty column for manual correction
          action = "",          # keep/remove/correct
          notes = ""
        )
      
      writexl::write_xlsx(review_df, review_file)
      log_warn("  Review mode: {nrow(flagged)} words flagged for review")
      log_warn("  Review file: {review_file}")
      
      return(list(
        requires_review = TRUE,
        review_file = review_file,
        flagged_words = flagged,
        clean_tx = NULL,
        removed_stats = removed_stats,
        rule_violations = NULL
      ))
    }
  }
  # AUTO MODE: Accept all transcriptions, no filtering by confidence
  
  ################################################################################
  # EXTRACT RULE VIOLATIONS AS FEATURES (BEFORE REMOVAL)
  ################################################################################
  
  fillers_to_remove <- config$transcription$filters$remove_fillers
  
  # 1. UMs count (per prompt)
  ums_per_prompt <- tx %>%
    mutate(is_um = word_clean %in% fillers_to_remove) %>%
    group_by(participant_id, prompt) %>%
    summarise(ums_count = sum(is_um, na.rm = TRUE), .groups = "drop")
  
  # 2. Comments count (per prompt)
  comments_per_prompt <- tx %>%
    mutate(is_comment = grepl("\\[|\\]|\\*", word)) %>%
    group_by(participant_id, prompt) %>%
    summarise(comment_count = sum(is_comment, na.rm = TRUE), .groups = "drop")
  
  # 3. Repeated prompts (per prompt)
  rep_prompts_per_prompt <- tx %>%
    mutate(is_rep_prompt = (prompt == word_clean)) %>%
    group_by(participant_id, prompt) %>%
    summarise(rep_prompt = sum(is_rep_prompt, na.rm = TRUE), .groups = "drop")
  
  # 4. Repeated words within each prompt (per prompt)
  rep_words_per_prompt <- tx %>%
    group_by(participant_id, prompt) %>%
    mutate(is_repeat = duplicated(word_clean)) %>%
    summarise(rep_words_per_prompt = sum(is_repeat, na.rm = TRUE), .groups = "drop")
  
  # Combine rule violations
  rule_violations <- list(
    ums = ums_per_prompt,
    comments = comments_per_prompt,
    repeated_prompts = rep_prompts_per_prompt,
    repeated_words_per_prompt = rep_words_per_prompt
  )
  
  # Update removed stats
  removed_stats$ums <- sum(ums_per_prompt$ums_count)
  removed_stats$comments <- sum(comments_per_prompt$comment_count)
  removed_stats$repetitions <- sum(rep_prompts_per_prompt$rep_prompt) + 
    sum(rep_words_per_prompt$rep_words_per_prompt)
  
  ################################################################################
  # UNIVERSAL CLEANUP (all modes) - NOW REMOVE AFTER COUNTING
  ################################################################################
  
  # Remove fillers and acknowledgments
  tx_clean <- tx %>%
    dplyr::filter(
      !word_clean %in% fillers_to_remove,
      !word_clean %in% config$transcription$filters$remove_acknowledgments,
      !grepl("\\[|\\]|\\*", word)  # Remove bracketed content
    )
  
  # Remove punctuation if configured
  if (config$transcription$filters$remove_punctuation) {
    tx_clean <- tx_clean %>%
      mutate(word_clean = gsub("[[:punct:]]", "", word_clean))
  }
  
  # Remove repeated prompts
  tx_clean <- tx_clean %>%
    dplyr::filter(prompt != word_clean)
  
  # Remove repeated words per prompt
  tx_clean <- tx_clean %>%
    group_by(participant_id, prompt) %>%
    mutate(is_repeat = duplicated(word_clean)) %>%
    ungroup() %>%
    dplyr::filter(!is_repeat) %>%
    select(-is_repeat)
  
  # Final cleanup: use corrected word
  tx_clean <- tx_clean %>%
    mutate(response = word_clean) %>%
    select(participant_id, prompt, task_number,
           start, end, response, confidence) %>%
    mutate(task_type=case_when(nchar(prompt)==1~"COWAT",
                               task_number %in% as.character(c(1:20)) ~ "WAT",
                               T~""))
  
  removed_stats$total_removed <- nrow(tx) - nrow(tx_clean)
  
  log_debug("  Cleanup summary:")
  log_debug("    - Fillers removed: {removed_stats$ums}")
  log_debug("    - Repetitions removed: {removed_stats$repetitions}")
  log_debug("    - Comments removed: {removed_stats$comments}")
  log_debug("    - Total removed: {removed_stats$total_removed}")
  log_debug("    - Remaining valid responses: {nrow(tx_clean)}")
  log_debug("  Rule violations saved as features for {length(unique(tx_clean$prompt))} prompts")
  
  return(list(
    requires_review = FALSE,
    clean_tx = tx_clean,
    removed_stats = removed_stats,
    review_file = NULL,
    rule_violations = rule_violations  # RETURN THESE FOR FEATURE EXTRACTION!
  ))
}