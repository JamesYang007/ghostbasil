# get path of current file
# https://stackoverflow.com/a/61739930/9936090
stub <- function() {}
thisPath <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  if (length(grep("^-f$", cmdArgs)) > 0) {
    # R console option
    normalizePath(dirname(cmdArgs[grep("^-f", cmdArgs) + 1]))[1]
  } else if (length(grep("^--file=", cmdArgs)) > 0) {
    # Rscript/R console option
    scriptPath <- normalizePath(dirname(sub("^--file=", "", cmdArgs[grep("^--file=", cmdArgs)])))[1]
  } else if (Sys.getenv("RSTUDIO") == "1") {
    # RStudio
    dirname(rstudioapi::getSourceEditorContext()$path)
  } else if (is.null(attr(stub, "srcref")) == FALSE) {
    # 'source'd via R console
    dirname(normalizePath(attr(attr(stub, "srcref"), "srcfile")$filename))
  } else {
    stop("Cannot find file path")
  }
}

generate.data <- function(n=10000, p=1000, rho=0.1, k=10, amplitude=7.5, seed=1234)
{
    set.seed(1234)

    # Generate the variables from a multivariate normal distribution
    mu = rep(0,p)
    Sigma = toeplitz(rho^(0:(p-1)))
    X = matrix(rnorm(n*p),n) %*% chol(Sigma)
    
    # Generate the response from a logistic model and encode it as a factor.
    nonzero = sample(p, k)
    beta = amplitude * (1:p %in% nonzero) / sqrt(n)
    y<-X%*%beta+rnorm(n)
    
    A<-t(X)%*%X/n; r<-cor(X,y);
    list(A=A, r=r)
}

