library(devtools)
library(lassosum)
library(fdrtool)

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

# Load GhostBASIL
GhostBASIL.path <- paste(thisPath(), '/../R', sep='')
load_all(GhostBASIL.path)

set.seed(1234)

elnetR <- function(lambda1, lambda2=0, X, b, thr=1e-4,
                   trace=0, maxiter=10000,
                   blocks=NULL,
                   x=NULL) {
  stopifnot(length(b) == ncol(X))
  diag <- colSums(X^2)
  
  if(length(lambda2) > 1) {
    nlambda2 <- length(lambda2)
    for(i in 1:nlambda2) {
      result <- elnetR(lambda1, lambda2[i], X, b, thr,
                       trace, maxiter, x)
      result <- list(fit=result, lambda2=lambda2[i])
      if(i == 1) Result <- rep(result, nlambda2) else
        Result[i] <- result
      
    }
    return(Result)
  }
  
  order <- order(lambda1, decreasing = T)
  lambda1a <- lambda1[order]
  conv <- lambda1a * NA
  len <- length(b)
  beta <- matrix(NA, len, length(lambda1))
  pred <- matrix(NA, nrow(X), length(lambda1))
  loss <- rep(NA, length(lambda1))
  fbeta <- loss
  
  if(is.null(x)) x <- b * 0.0 else {
    stopifnot(length(x) == len)
    x <- x + 0.0 # Making sure R creates a copy...
  }
  
  if(is.null(blocks)) {
    Blocks <- list(startvec=0, endvec=len - 1)
  } else {
    Blocks <- parseblocks(blocks)
    stopifnot(max(Blocks$endvec)==len - 1)
  }
  
  X <- as.matrix(X)
  yhat <- as.vector(X %*% x)
  
  for(i in 1:length(lambda1a)) {
    if(trace > 0) cat("lambda1: ", lambda1a[i], "\n")
    conv[i] <- repelnet(lambda1a[i], lambda2, diag, X, b,thr,x,yhat, trace-1,maxiter,
                        Blocks$startvec, Blocks$endvec)
    if(conv[i] != 1) stop("Not converging...")
    
    beta[,i] <- x
    pred[,i] <- yhat
    loss[i] <- sum(yhat^2) - 2* sum(b * x)
    fbeta[i] <- loss[i] + 2* sum(abs(x))*lambda1a[i] + sum(x^2)*lambda2
  }
  
  conv[order] <- conv
  beta[,order] <- beta
  pred[,order] <- pred
  loss[order] <- loss
  fbeta[order] <- fbeta
  
  return(list(lambda1=lambda1, lambda2=lambda2, beta=beta, conv=conv, pred=pred, loss=loss, fbeta=fbeta))
}

repelnet <- function(lambda1, lambda2, diag, X, r, thr, x, yhat, trace, maxiter, startvec, endvec) {
  .Call('_lassosum_repelnet', PACKAGE = 'lassosum', lambda1, lambda2, diag, X, r, thr, x, yhat, trace, maxiter, startvec, endvec)
}


solve.BASIL<-function(A,r,s=0.5,M=100,dM=50,maxiter=1000,cbound=10000){
  
  p<-length(r)
  epsilon <- .0001
  K <- 100
  lambda<-c();beta<-c()
  
  #BASIL step 1
  abs_Df<-2*abs(r)
  include.index<-order(abs_Df,decreasing=T)[1:M]
  remaining.index<-(1:p)[-include.index]
  
  lambda_max<-max(abs_Df)
  lambdapath <- round(exp(seq(log(lambda_max), log(lambda_max*epsilon),
                              length.out = K)), digits = 10)
  #lambdapath
  
  refpanel<-chol(A[include.index,include.index,drop=F])
  fit.lassosum<-elnetR(lambda1=lambdapath,lambda2=s,X=refpanel,b=r[include.index],maxiter=maxiter)
  
  fit.lassosum$lambda1
  fit.lassosum$beta
  beta_update<-matrix(0,p,length(fit.lassosum$lambda1))
  beta_update[include.index,]<-fit.lassosum$beta
  abs_Df_update<-abs(2*A%*%beta_update*(1-s)-2*as.vector(r)+2*s*beta_update)
  lambda_max_update<-min(fit.lassosum$lambda1[which(apply(abs_Df_update[remaining.index,],2,max)<fit.lassosum$lambda1)])
  
  #update results
  lambda<-cbind(lambda,fit.lassosum$lambda1[fit.lassosum$lambda1>=lambda_max_update])
  beta<-cbind(beta,beta_update[,fit.lassosum$lambda1>=lambda_max_update])
  
  abs_Df<-abs_Df_update
  abs_Df_remaining<-abs_Df[remaining.index,sum(fit.lassosum$lambda1>=lambda_max_update)]
  
  lambda_max<-max(abs_Df_remaining)
  include.index<-c(include.index,remaining.index[order(abs_Df_remaining,decreasing=T)[1:min(dM,length(remaining.index))]])
  remaining.index<-(1:p)[-include.index]
  
  #options(warn = 0)
  #BASIL step k
  if(max(0,(cbound-M))/dM+1>=2){
    for(k in 2:(max(0,(cbound-M))/dM+1)){
      #print(k)
      #print(lambda)
      
      lambdapath <- round(exp(seq(log(lambda_max), log(lambda_max*epsilon),
                                  length.out = K)), digits = 10)
      #lambdapath
      
      refpanel<-chol(A[include.index,include.index,drop=F])
      fit.lassosum<-try(elnetR(lambda1=lambdapath,lambda2=s,X=refpanel,b=r[include.index],maxiter=maxiter),silent=T)
      if(class(fit.lassosum)=='try-error'){break}
      #fit.lassosum$lambda1
      #fit.lassosum$beta
      beta_update<-matrix(0,p,length(fit.lassosum$lambda1))
      beta_update[include.index,]<-fit.lassosum$beta
      if(length(remaining.index)!=0){
        abs_Df_update<-abs(2*A%*%beta_update*(1-s)-2*as.vector(r)+2*s*beta_update)
        index.KKT<-which(apply(abs_Df_update[remaining.index,],2,max)<fit.lassosum$lambda1)
        if(length(index.KKT)==0){
          lambda_max<-lambda[length(lambda)]
          abs_Df_remaining<-abs_Df[remaining.index,ncol(abs_Df)]
          include.index<-c(include.index,remaining.index[order(abs_Df_remaining,decreasing=T)[1:min(dM,length(remaining.index))]])
          remaining.index<-(1:p)[-include.index]
          next
        }
        lambda_max_update<-min(fit.lassosum$lambda1[which(apply(abs_Df_update[remaining.index,],2,max)<fit.lassosum$lambda1)])
        lambda<-c(lambda,fit.lassosum$lambda1[fit.lassosum$lambda1>=lambda_max_update])
        beta<-cbind(beta,beta_update[,fit.lassosum$lambda1>=lambda_max_update])
        
        abs_Df<-abs_Df_update
        abs_Df_remaining<-abs_Df[remaining.index,sum(fit.lassosum$lambda1>=lambda_max_update)]
        
        lambda_max<-max(abs_Df_remaining)
        include.index<-c(include.index,remaining.index[order(abs_Df_remaining,decreasing=T)[1:min(dM,length(remaining.index))]])
        remaining.index<-(1:p)[-include.index]
      }else{
        lambda<-c(lambda,fit.lassosum$lambda1)
        beta<-cbind(beta,beta_update)
        break
      }
    }
  }
  
  return(list(lambda=lambda,beta=beta))
}



solve.BASIL.PV<-function(A,r,n,s=0.5,M=100,dM=50,maxiter=1000,cbound=10000){
  p<-length(r)
  epsilon <- .0001
  K <- 100
  lambda<-c();beta<-c();f.lambda<-c()
  
  #local FDR, shrunken estimate
  fdr <- fdrtool::fdrtool(as.vector(r)*sqrt(n), statistic="normal",
                          plot=F)
  r.shrunk <- as.matrix(r * (1 - fdr$lfdr))
  
  #BASIL step 1
  abs_Df<-2*abs(r)
  include.index<-order(abs_Df,decreasing=T)[1:M]
  remaining.index<-(1:p)[-include.index]
  
  lambda_max<-max(abs_Df)
  lambdapath <- round(exp(seq(log(lambda_max), log(lambda_max*epsilon),
                              length.out = K)), digits = 10)
  #lambdapath
  
  refpanel<-chol(A[include.index,include.index,drop=F])
  fit.lassosum<-elnetR(lambda1=lambdapath,lambda2=s,X=refpanel,b=r[include.index],maxiter=maxiter)
  
  #fit.lassosum$lambda1
  #fit.lassosum$beta
  beta_update<-matrix(0,p,length(fit.lassosum$lambda1))
  beta_update[include.index,]<-fit.lassosum$beta
  abs_Df_update<-abs(2*A%*%beta_update*(1-s)-2*as.vector(r)+2*s*beta_update)
  lambda_max_update<-min(fit.lassosum$lambda1[which(apply(abs_Df_update[remaining.index,],2,max)<fit.lassosum$lambda1)])
  update.index<-which(fit.lassosum$lambda1>=lambda_max_update)
  beta_update<-beta_update[,update.index,drop=F]
  
  #update results
  lambda<-c(lambda,fit.lassosum$lambda1[update.index])
  beta<-cbind(beta,beta_update)
  
  #Get.f<-function(x){x<-as.matrix(x);return(t(x)%*%r.shrunk/sqrt(t(x)%*%A%*%x))}
  f.lambda.update<-as.vector(t(beta_update)%*%r.shrunk/sqrt(colSums(beta_update*A%*%beta_update)))
  f.lambda.update[is.na(f.lambda.update)]<-0
  f.lambda<-c(f.lambda,f.lambda.update)
  
  abs_Df<-abs_Df_update
  abs_Df_remaining<-abs_Df[remaining.index,sum(fit.lassosum$lambda1>=lambda_max_update)]
  
  lambda_max<-max(abs_Df_remaining)
  include.index<-c(include.index,remaining.index[order(abs_Df_remaining,decreasing=T)[1:min(dM,length(remaining.index))]])
  remaining.index<-(1:p)[-include.index]
  
  #options(warn = 0)
  #BASIL step k
  if(max(0,(cbound-M))/dM+1>=2){
    for(k in 2:(max(0,(cbound-M))/dM+1)){
      #print(k)
      #print(lambda)
      
      lambdapath <- round(exp(seq(log(lambda_max), log(lambda_max*epsilon),
                                  length.out = K)), digits = 10)
      #lambdapath
      
      refpanel<-chol(A[include.index,include.index,drop=F])
      fit.lassosum<-try(elnetR(lambda1=lambdapath,lambda2=s,X=refpanel,b=r[include.index],maxiter=maxiter),silent=T)
      if(class(fit.lassosum)=='try-error'){break}
      #fit.lassosum$lambda1
      #fit.lassosum$beta
      beta_update<-matrix(0,p,length(fit.lassosum$lambda1))
      beta_update[include.index,]<-fit.lassosum$beta
      if(length(remaining.index)!=0){
        abs_Df_update<-abs(2*A%*%beta_update*(1-s)-2*as.vector(r)+2*s*beta_update)
        index.KKT<-which(apply(abs_Df_update[remaining.index,],2,max)<fit.lassosum$lambda1)
        if(length(index.KKT)==0){
          lambda_max<-lambda[length(lambda)]
          abs_Df_remaining<-abs_Df[remaining.index,ncol(abs_Df)]
          include.index<-c(include.index,remaining.index[order(abs_Df_remaining,decreasing=T)[1:min(dM,length(remaining.index))]])
          remaining.index<-(1:p)[-include.index]
          next
        }
        lambda_max_update<-min(fit.lassosum$lambda1[which(apply(abs_Df_update[remaining.index,],2,max)<fit.lassosum$lambda1)])
        update.index<-which(fit.lassosum$lambda1>=lambda_max_update)
        beta_update<-beta_update[,update.index,drop=F]
        
        #stopping rule
        #Get.f<-function(x){x<-Matrix(as.matrix(x));return(t(x)%*%r.shrunk/sqrt(t(x)%*%A%*%x))}
        #f.lambda.update<-apply(beta_update,2,Get.f)
        f.lambda.update<-as.vector(t(beta_update)%*%r.shrunk/sqrt(colSums(beta_update*A%*%beta_update)))
        f.lambda.update[is.na(f.lambda.update)]<-0
        stop.index<-max(f.lambda.update,na.rm=T)<max(f.lambda,na.rm=T)
        if(stop.index==T){break}
        
        #update results
        lambda<-c(lambda,fit.lassosum$lambda1[update.index])
        beta<-cbind(beta,beta_update)
        f.lambda<-c(f.lambda,f.lambda.update)
        
        abs_Df<-abs_Df_update
        abs_Df_remaining<-abs_Df[remaining.index,sum(fit.lassosum$lambda1>=lambda_max_update)]
        
        lambda_max<-max(abs_Df_remaining)
        include.index<-c(include.index,remaining.index[order(abs_Df_remaining,decreasing=T)[1:min(dM,length(remaining.index))]])
        remaining.index<-(1:p)[-include.index]
      }else{
        lambda<-c(lambda,fit.lassosum$lambda1)
        beta<-cbind(beta,beta_update)
        f.lambda<-apply(beta,2,Get.f)
        f.lambda[is.na(f.lambda)]<-0
        break
      }
    }
    #f.lambda<-apply(beta,2,Get.f)
    #f.lambda[is.na(f.lambda)]<-0
    beta.final<-beta[,which.max(f.lambda),drop=F]
    lambda.final<-lambda[which.max(f.lambda)]
  }
  
  return(list(lambda=lambda,beta=beta,f.lambda=f.lambda,beta.final=beta.final,lambda.final=lambda.final))
}

# Computes objective value for BASIL and ghostbasil methods
# at the same lambda value at index idx.test.
# We want ghostbasil objective to be <= BASIL objective.
# First, run speed.test and get output (out).
# Usage example: obj.test(out$fits[[1]], 15)
# means for the first p size, test objective at 15th lambda.
obj.test <- function(fit, idx.test=1)
{
    A <- fit[['A']]
    r <- fit[['r']]
    s <- fit[['s']]
    fit.BASIL <- fit[['BASIL']]
    fit.lassosum <- fit[['CD']]
    fit.BASIL.PV <- fit[['BASIL.PV']]
    fit.ghostbasil <- fit[['ghostbasil']]

    ghostbasil.lmda.test <- fit.ghostbasil$lmdas[idx.test]
    
    if (((length(fit.BASIL$lambda) >= idx.test) & (ghostbasil.lmda.test != fit.BASIL$lambda[idx.test])) |
        ((length(fit.lassosum$lambda1) >= idx.test) & (ghostbasil.lmda.test != fit.lassosum$lambda1[idx.test])) |
        ((length(fit.BASIL.PV$lambda) >= idx.test) & (ghostbasil.lmda.test != fit.BASIL.PV$lambda[min(idx.test)])) ) {
        stop("Lambda values do not match.")
    }

    BASIL.beta.test <- if (ncol(fit.BASIL$beta) < idx.test) NA else fit.BASIL$beta[,idx.test,drop=F]
    cd.beta.test <- if (ncol(fit.lassosum$beta) < idx.test) NA else fit.lassosum$beta[,idx.test,drop=F]
    BASIL.PV.beta.test <- if (ncol(fit.BASIL.PV$beta) < idx.test) NA else fit.BASIL.PV$beta[,idx.test,drop=F]
    ghostbasil.beta.test <- if (ncol(fit.ghostbasil$beta) < idx.test) NA else fit.ghostbasil$betas[,idx.test,drop=F]

    BASIL.objective <- if (any(is.na(ghostbasil.beta.test))) NA else objective(A, r, s, ghostbasil.lmda.test, BASIL.beta.test)
    cd.objective <- if (any(is.na(cd.beta.test))) NA else objective(A, r, s, ghostbasil.lmda.test, cd.beta.test)
    BASIL.PV.objective <- if (any(is.na(BASIL.PV.beta.test))) NA else objective(A, r, s, ghostbasil.lmda.test, BASIL.PV.beta.test)
    ghostbasil.objective <- if (any(is.na(ghostbasil.beta.test))) NA else objective(A, r, s, ghostbasil.lmda.test, ghostbasil.beta.test)

    objs <- c(BASIL=BASIL.objective, 
              CoordinateDescent=cd.objective,
              BASIL.PV=BASIL.PV.objective,
              ghostbasil=ghostbasil.objective)
    objs
}

# Checks KKT for all methods in speed.test
# at the same lambda value at index idx.test.
# We want ghostbasil to be TRUE more often than the other methods.
# Because of numerical precision, the KKT condition must be relaxed with some slack.
# This checks if the absolute gradient is <= lmda + eps for all features.
#
# First, run speed.test and get output (out).
# Usage example: kkt.test(out$fits[[1]], 15)
# means for the first p size, check KKT at 15th lambda.
kkt.test <- function(fit, idx.test=1, eps=1e-5)
{
    A <- fit[['A']]
    r <- fit[['r']]
    s <- fit[['s']]
    fit.BASIL <- fit[['BASIL']]
    fit.lassosum <- fit[['CD']]
    fit.BASIL.PV <- fit[['BASIL.PV']]
    fit.ghostbasil <- fit[['ghostbasil']]

    ghostbasil.lmda.test <- fit.ghostbasil$lmdas[idx.test]

    if (((length(fit.BASIL$lambda) >= idx.test) & (ghostbasil.lmda.test != fit.BASIL$lambda[idx.test])) |
        ((length(fit.lassosum$lambda1) >= idx.test) & (ghostbasil.lmda.test != fit.lassosum$lambda1[idx.test])) |
        ((length(fit.BASIL.PV$lambda) >= idx.test) & (ghostbasil.lmda.test != fit.BASIL.PV$lambda[min(idx.test)])) ) {
        stop("Lambda values do not match.")
    }

    BASIL.beta.test <- if (ncol(fit.BASIL$beta) < idx.test) NA else fit.BASIL$beta[,idx.test,drop=F]
    cd.beta.test <- if (ncol(fit.lassosum$beta) < idx.test) NA else fit.lassosum$beta[,idx.test,drop=F]
    BASIL.PV.beta.test <- if (ncol(fit.BASIL.PV$beta) < idx.test) NA else fit.BASIL.PV$beta[,idx.test,drop=F]
    ghostbasil.beta.test <- if (ncol(fit.ghostbasil$beta) < idx.test) NA else fit.ghostbasil$betas[,idx.test,drop=F]

    check.kkt <- function(A, r, s, lmda, beta) {
        sum(abs((1-s) * (A %*% beta) - r + s * beta) > lmda+eps) == 0
    }

    BASIL.check.kkt <- if (any(is.na(ghostbasil.beta.test))) NA else check.kkt(A, r, s, ghostbasil.lmda.test, BASIL.beta.test)
    cd.check.kkt <- if (any(is.na(cd.beta.test))) NA else check.kkt(A, r, s, ghostbasil.lmda.test, cd.beta.test)
    BASIL.PV.check.kkt <- if (any(is.na(BASIL.PV.beta.test))) NA else check.kkt(A, r, s, ghostbasil.lmda.test, BASIL.PV.beta.test)
    ghostbasil.check.kkt <- if (any(is.na(ghostbasil.beta.test))) NA else check.kkt(A, r, s, ghostbasil.lmda.test, ghostbasil.beta.test)

    out <- c(BASIL=BASIL.check.kkt, 
              CoordinateDescent=cd.check.kkt,
              BASIL.PV=BASIL.PV.check.kkt,
              ghostbasil=ghostbasil.check.kkt)
    out
}

speed.test <- function(ps=seq(1000, 3000, by=1000),
                       n=10000, # number of observations
                       k=10,# number of variables with nonzero coefficients
                       amplitude=7.5,# signal amplitude (for noise level = 1)
                       rho=0.1,
                       s=0.5)
{
    objectives <- list()
    kkts <- list()
    cpu.time<-c()

    for (p in ps) {
      print(p)
      
      # Generate the variables from a multivariate normal distribution
      mu = rep(0,p)
      Sigma = toeplitz(rho^(0:(p-1)))
      X = matrix(rnorm(n*p),n) %*% chol(Sigma)
      
      # Generate the response from a logistic model and encode it as a factor.
      nonzero = sample(p, k)
      beta = amplitude * (1:p %in% nonzero) / sqrt(n)
      y<-X%*%beta+rnorm(n)
      
      A<-t(X)%*%X/n; r<-cor(X,y);
      
      t1<-proc.time()
      fit.BASIL<-solve.BASIL(A,r,s=s,M=500,dM=200,maxiter=1000,cbound=1000)
      t2<-proc.time()
      temp.time<-t2[3]-t1[3]
      
      #verify results
      t1<-proc.time()
      temp.X<-chol(A)
      fit.lassosum<-elnetR(lambda1=fit.BASIL$lambda,lambda2=s,X=temp.X,b=r,maxiter=1000)
      t2<-proc.time()
      temp.time<-c(temp.time,t2[3]-t1[3])
      
      t1<-proc.time()
      fit.BASIL.PV<-solve.BASIL.PV(A,r,n=n,s=s,M=500,dM=200,maxiter=1000,cbound=1000)
      t2<-proc.time()
      temp.time<-c(temp.time,t2[3]-t1[3])
      
      t1<-proc.time()
      fit.ghostbasil <- ghostbasil(A,r,s,
                                   user.lambdas=fit.BASIL$lambda,
                                   lambdas.iter=10,
                                   strong.size=500,
                                   delta.strong.size=200,
                                   max.strong.size=p,
                                   max.cds=1000)
      t2<-proc.time()
      temp.time<-c(temp.time,t2[3]-t1[3])
      
      cpu.time<-rbind(cpu.time,temp.time)
      fit <- list(A=A, r=r, s=s, BASIL=fit.BASIL, CD=fit.lassosum, BASIL.PV=fit.BASIL.PV, ghostbasil=fit.ghostbasil)
      fit.objective <- do.call(rbind, lapply(1:length(fit.BASIL$lambda), function(i) t(data.frame(obj.test(fit, i)))))
      fit.kkt <- do.call(rbind, lapply(1:length(fit.BASIL$lambda), function(i) t(data.frame(kkt.test(fit, i)))))
      rownames(fit.objective) <- NULL
      rownames(fit.kkt) <- NULL
      objectives <- append(objectives, list(fit.objective))
      kkts <- append(kkts, list(fit.kkt))
      print(cpu.time)
    }
    result<-cbind(ps, cpu.time)
    colnames(result)<-c('p','BASIL','CoordinateDescent','BASIL-PV','ghostbasil')
    rownames(result)<-NULL
    list(times=result, objectives=objectives, kkts=kkts)
}

