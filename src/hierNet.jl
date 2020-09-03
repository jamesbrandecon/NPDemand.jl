function hierNet(s::Matrix, p::Matrix, iv::Matrix, nfolds::Integer, nlam::Integer, strong::Bool)
J = size(s,2);
@rput s
@rput p
@rput iv
@rput strong
@rput nlam
@rput nfolds
@rput J
R"
library(hierNet)
J <-  ncol(s)

rownames(s) <- seq(length=nrow(s))
rownames(p) <- seq(length=nrow(p))
rownames(iv) <- seq(length=nrow(iv))

# Instrument for market shares
s_hat <- matrix(nrow = nrow(s),ncol = ncol(s))
one_mat <- matrix(1,nrow(s), 1)
iv <- as.matrix(iv)
for (j in 1:J) {
  iv_temp <- as.matrix(cbind(matrix(1,nrow(s),1),iv))
  beta<- solve(t(iv_temp)%*%iv_temp) %*% t(iv_temp) %*% s[,j]
  s_hat[,j] <-iv_temp%*% beta
}

# Estimate Hierarchical Lasso
for (j in 1:J) {
   # Estimate path of regularization parameters
  structured_fit <- hierNet.path(as.matrix(s_hat), as.vector(p[,j]),
                                lamlist = NULL, delta=1e-8, minlam = NULL, maxlam = NULL, nlam=nlam, flmin=.05,
                                diagonal = TRUE, strong = strong, aa = NULL, zz = NULL,
                                stand.main = TRUE, stand.int = TRUE,
                                rho = nrow(s), niter = 100, sym.eps = 0.001,
                                step = 1, maxiter = 2000, backtrack = 0.2, tol = 1e-05, trace = 0)
  if (j==1) {
    matsize <- nrow(structured_fit$bp)
    coefs <- matrix(nrow = 0 ,ncol = matsize)
  }

  # Select parameter by CV, append to matrix
  a<- hierNet.cv(structured_fit, as.matrix(s_hat), as.vector(p[,j]), nfolds=nfolds,folds=NULL,trace=0)
  lambdahat <- a['lamhat.1se']
  # lamhat lamhat.1se

  # Final Estimates
  structured_fit<- hierNet(as.matrix(s_hat), as.vector(p[,j]), as.numeric(lambdahat), delta=1e-8, strong=strong, diagonal=TRUE, aa=NULL,
                             zz=NULL,center=TRUE, stand.main=TRUE, stand.int=TRUE, rho=nrow(s), niter=100,
                             sym.eps=1e-3,step=1, maxiter=5000, backtrack=0.2, tol=1e-5, trace=0)
  coefs <- rbind(coefs,t(structured_fit$bp - structured_fit$bn))
}
included = (abs(coefs)>0)

for (j in 1:J) {
  included[j,j] = 1
}

included_symmetric <- included
for (j1 in 1:J) {
  for (j2 in 1:J) {
    if (included_symmetric[j1,j2] != included_symmetric[j2,j1]){
      included_symmetric[j1,j2] = 1
      included_symmetric[j2,j1] = 1
    }
  }
}
"
@rget included
@rget included_symmetric
return included, included_symmetric
end
