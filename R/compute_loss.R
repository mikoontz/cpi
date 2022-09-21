
# Internal function to compute sample loss
compute_loss <- function(pred, measure, response_is_prob = FALSE) {
  if (inherits(pred, "Prediction")) {
    truth <- pred$truth
    response <- pred$response
    prob <- pred$prob
  } else {
    truth <- do.call(c, lapply(pred, function(x) x$truth))
    response <- do.call(c, lapply(pred, function(x) x$response))
    prob <- do.call(rbind, lapply(pred, function(x) x$prob))
  }
  
  if (response_is_prob) {
    prob <- response
  }
  
  if (measure$id == "regr.mse") {
    # Squared errors
    loss <- (truth - response)^2
  } else if (measure$id == "regr.mae") {
    # Absolute errors
    loss <- abs(truth - response)
  } else if (measure$id == "classif.logloss") {
    # Logloss 
    eps <- 1e-15
    if (response_is_prob) {
      prob <- setNames(data.frame(1 - prob, prob), nm = c("0", "1"))
    }
    ii <- match(as.character(truth), colnames(prob))
    p <- prob[cbind(seq_len(nrow(prob)), ii)]
    p <- pmax(eps, pmin(1 - eps, p))
    loss <- -log(p)
  } else if (measure$id == "classif.ce") {
    # Misclassification error
    loss <- 1*(truth != response)
  } else if (measure$id == "classif.bbrier") {
    # Brier score
    # First level is positive class
    y <- as.numeric(as.numeric(truth) == 1)
    loss <- (y - prob[, 1])^2
  } else if (measure$id == "classif.fbeta") {
    
    if (response_is_prob) {
      response_class <- ifelse(prob >= 0.5, yes = 1, no = 0)
    } else {
      response_class <- response
    }
    
    tp <- length(which(truth == 1 & response_class == 1))
    fp <- length(which(truth == 0 & response_class == 1))
    fn <- length(which(truth == 1 & response_class == 0))
    
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    f1 <- 2 * (precision * recall / (precision + recall))
    
    loss <- 1 - f1
    
  } else if (measure$id == "classif.mcc") {
    
    if (response_is_prob) {
      response_class <- ifelse(prob >= 0.5, yes = 1, no = 0)
    } else {
      response_class <- response
    }
    
    tp <- as.numeric(length(which(truth == 1 & response_class == 1)))
    fp <- as.numeric(length(which(truth == 0 & response_class == 1)))
    tn <- as.numeric(length(which(truth == 0 & response_class == 0)))
    fn <- as.numeric(length(which(truth == 1 & response_class == 0)))
    
    mcc <- ((tp * tn) - (fn * fp)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    loss <- -1 * mcc  
    
  } else {
    stop("Unknown measure.")
  }
  
  loss
}

