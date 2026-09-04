args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  parsed <- list(
    `bed-prefix` = NULL,
    `out-prefix` = NULL,
    `num-pcs` = "10",
    `threads` = "8"
  )

  idx <- 1
  while (idx <= length(args)) {
    key <- sub("^--", "", args[[idx]])
    if (!(key %in% names(parsed))) {
      stop(sprintf("Unknown argument: %s", args[[idx]]), call. = FALSE)
    }
    if (idx == length(args)) {
      stop(sprintf("Missing value for argument: %s", args[[idx]]), call. = FALSE)
    }
    parsed[[key]] <- args[[idx + 1]]
    idx <- idx + 2
  }

  if (is.null(parsed[["bed-prefix"]]) || is.null(parsed[["out-prefix"]])) {
    stop("--bed-prefix and --out-prefix are required.", call. = FALSE)
  }

  parsed
}

opts <- parse_args(args)

suppressPackageStartupMessages(library(SNPRelate))

bed_prefix <- opts[["bed-prefix"]]
out_prefix <- opts[["out-prefix"]]
num_pcs <- as.integer(opts[["num-pcs"]])
threads <- as.integer(opts[["threads"]])

bed_path <- paste0(bed_prefix, ".bed")
bim_path <- paste0(bed_prefix, ".bim")
fam_path <- paste0(bed_prefix, ".fam")
gds_path <- paste0(out_prefix, ".gds")
rds_path <- paste0(out_prefix, ".rds")
pc_csv_path <- paste0(out_prefix, "_pc.csv")

dir.create(dirname(gds_path), recursive = TRUE, showWarnings = FALSE)

SNPRelate::snpgdsBED2GDS(
  bed.fn = bed_path,
  bim.fn = bim_path,
  fam.fn = fam_path,
  out.gdsfn = gds_path,
  cvt.chr = "int",
  cvt.snpid = "auto",
  verbose = TRUE
)

genofile <- SNPRelate::snpgdsOpen(gds_path)
on.exit(SNPRelate::snpgdsClose(genofile), add = TRUE)

pca <- SNPRelate::snpgdsPCA(
  genofile,
  num.thread = threads,
  remove.monosnp = TRUE,
  autosome.only = FALSE,
  verbose = TRUE
)

saveRDS(pca, file = rds_path)

fam <- utils::read.table(fam_path, header = FALSE, stringsAsFactors = FALSE)
colnames(fam) <- c("ID_1", "ID_2", "father", "mother", "sex", "phenotype")

pc_df <- data.frame(sample.id = pca$sample.id, stringsAsFactors = FALSE)
fam_idx <- match(pc_df$sample.id, fam$ID_2)
if (any(is.na(fam_idx))) {
  if (nrow(fam) != nrow(pc_df)) {
    stop("Could not align SNPRelate sample IDs to the PLINK FAM file.", call. = FALSE)
  }
  fam_idx <- seq_len(nrow(fam))
}

pc_df$ID_1 <- fam$ID_1[fam_idx]
pc_df$ID_2 <- fam$ID_2[fam_idx]
pc_df$IID <- pc_df$ID_2

pcs_to_write <- min(num_pcs, ncol(pca$eigenvect))
for (idx in seq_len(pcs_to_write)) {
  pc_df[[sprintf("PC%d", idx)]] <- pca$eigenvect[, idx]
}

utils::write.csv(pc_df, file = pc_csv_path, row.names = FALSE, quote = TRUE)
