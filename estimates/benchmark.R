# The benchmark referees will ask for: eiPack::ei.MD.bayes, the hierarchical
# multinomial-Dirichlet R x C model of Rosen, Jiang, King & Tanner (2001).
#
# Called by 11_benchmark.py, which prepares the CSV and scores the output. Reads a
# saluran-level table of row margins (registered by group) and column margins (choice
# counts), fits the MD model, and writes the implied group x choice rates.
#
# This is the standard against which our pipeline should be judged: it parameterises the
# internal cells directly, so it respects the deterministic bounds by construction -- the
# same property we obtain by raking -- but it does so inside a coherent Bayesian model
# rather than by a two-step seed-then-reconcile procedure.
#
# Usage: Rscript 11_eipack.R <in.csv> <out.csv> <n_rows> <n_cols> <iters> <burn> <thin>

suppressMessages(library(eiPack))

args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; outfile <- args[2]

# ei.MD.bayes with ret.beta="s" writes ONE FILE PER BETA into the working directory --
# 57,120 of them for a 2,856-unit 4x5 table, dumped wherever R happens to be running.
# We use ret.beta="d" (discard): only the aggregate Cell.counts draws are needed below, and
# retaining per-unit betas for a 6x5 table over a long chain exhausts R's vector memory
# limit (32 Gb) outright. Run from a scratch directory regardless, so a future change of
# that flag cannot litter the repo.
scratch <- file.path(dirname(outfile), "eipack_scratch")
dir.create(scratch, showWarnings = FALSE, recursive = TRUE)
setwd(scratch)
nr <- as.integer(args[3]); nc <- as.integer(args[4])
iters <- as.integer(args[5]); burn <- as.integer(args[6]); thin <- as.integer(args[7])

d <- read.csv(infile, check.names = FALSE)
rows <- paste0("r", seq_len(nr))
cols <- paste0("c", seq_len(nc))

# ei.MD.bayes wants counts on both margins and a total that reconciles them; the caller
# guarantees rowsum == colsum == electorate for every saluran.
form <- as.formula(paste0("cbind(", paste(cols, collapse = ","), ") ~ cbind(",
                          paste(rows, collapse = ","), ")"))

cat("fitting ei.MD.bayes:", nrow(d), "units,", nr, "x", nc,
    "table,", iters, "iters\n")
t0 <- Sys.time()
fit <- ei.MD.bayes(form, data = d, total = "total",
                   sample = iters, burnin = burn, thin = thin,
                   ret.beta = "d", ret.mcmc = TRUE, verbose = 0)
cat("elapsed:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")

# Cell.counts holds the aggregate row x column counts per posterior draw, summed over
# units -- exactly our estimand once divided by the row total. lambda.MD() would give the
# same thing but errors on this fit object, and going via the cell counts is anyway more
# transparent: it is verifiably the same accounting as everything else here (the counts
# sum to the electorate exactly).
cc <- fit$draws$Cell.counts
lam <- t(apply(cc, 1, function(x) {
  m <- matrix(x, nrow = nr, ncol = nc,
              dimnames = list(rows, cols))          # ccount.rR.cC, row index fastest
  as.vector(m / rowSums(m))
}))
colnames(lam) <- as.vector(outer(rows, cols, function(a, b) paste0(a, ".", b)))

out <- data.frame(param = colnames(lam),
                  mean = apply(lam, 2, mean),
                  sd = apply(lam, 2, sd))
write.csv(out, outfile, row.names = FALSE)
cat("wrote", outfile, "\n")
