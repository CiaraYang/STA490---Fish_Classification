files <- list.files(path = ".", pattern = "\\.[Rr]$", recursive = TRUE, full.names = TRUE)

extract_pkgs <- function(lines) {
  pkgs1 <- sub(".*library\\(([^)]+)\\).*", "\\1",
               grep("library\\(", lines, value = TRUE))
  pkgs2 <- sub(".*require\\(([^)]+)\\).*", "\\1",
               grep("require\\(", lines, value = TRUE))
  pkgs3 <- sub("['\"]", "", unlist(regmatches(
    lines,
    gregexpr("[A-Za-z0-9.]+(?=::)", lines, perl = TRUE)
  )))

  pkgs <- c(pkgs1, pkgs2, pkgs3)
  pkgs <- gsub("['\" ]", "", pkgs)
  pkgs[nzchar(pkgs)]
}

all_pkgs <- unique(unlist(lapply(files, function(f) {
  lines <- tryCatch(readLines(f, warn = FALSE), error = function(e) character())
  extract_pkgs(lines)
})))

all_pkgs <- sort(all_pkgs)
cat(paste(all_pkgs, collapse = "\n"))
cat("\n")

