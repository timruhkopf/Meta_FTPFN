Yeo-Johnson, explained                                                                                                  
                                                                                                                          
  Box-Cox first, since Yeo-Johnson is a direct generalization of it. Box-Cox is the classical "power transform" family for
  straightening out skewed positive data:                                                                                 
                                                                                                                          
  $$ \text{BC}(y;\lambda) = \begin{cases} \dfrac{y^\lambda - 1}{\lambda} & \lambda \neq 0 \ \log y & \lambda = 0          
  \end{cases} $$                                                                                                          
                                                                                                                          
  At $\lambda=1$ it's (an affine reparametrization of) the identity; at $\lambda=0$ it's $\log$ — the classic fix for a   
  long right tail (occasional huge values), which is a very real HPO phenomenon (a handful of divergent configs with      
  enormous loss, exactly what we found breaking TaskSet's fit earlier — Box-Cox/Yeo-Johnson would have been the right tool
  there too). At $\lambda=2$ it's $y^2$, the opposite correction, for left-skew. It requires $y>0$ strictly, which is a   
  real limitation — many HPO metrics (log-likelihood, margins) can be negative.                                           
                                                                                                                          
  Yeo-Johnson (Yeo & Johnson, 2000) extends this to the whole real line by applying Box-Cox-like power transforms         
  separately to the non-negative and negative parts, chosen so the two halves join up smoothly at zero:                   
                                                                                                                          
  $$ \psi(y;\lambda) = \begin{cases} \dfrac{(y+1)^\lambda - 1}{\lambda} & y \ge 0,\ \lambda \neq 0 \ \log(y+1) & y \ge 0,\
  \lambda = 0 \ -\dfrac{(-y+1)^{2-\lambda} - 1}{2-\lambda} & y < 0,\ \lambda \neq 2 \ -\log(-y+1) & y < 0,\ \lambda = 2   
  \end{cases} $$                                                                                                          
                                                                                                                          
  Two design choices worth understanding, not just accepting:                                                             
                                                                                                                          
  - The +1/-1 shifts: applying the power to $(y+1)$ rather than $y$ avoids the singularity a non-integer power would have 
    at $y=0$, and guarantees $\psi(0;\lambda)=0$ for every $\lambda$ — the two branches meet continuously at the origin   
    regardless of $\lambda$.                                                                                              
  - The mirrored exponent $(2-\lambda)$ on the negative branch: chosen specifically so that $\lambda=1$ collapses both    
    branches to the exact identity — check it: for $y\ge0$, $((y+1)^1-1)/1 = y$; for $y<0$, $-((-y+1)^{(2-1)}-1)/(2-1) =  
    -(-y) = y$. Verified numerically before committing to this. This nesting is exactly why it's the right building block:
    "no shape distortion, just affine" is one specific, exactly-reachable point ($\lambda=1$) in the family, not an       
    approximation.                                                                                                        
                                                                                                                          
  Why this specific family is the right fit for HPO y-distortion (not just "a" constrained family): it has exactly one    
  shape parameter beyond scale/shift, which matches a real domain fact — boundedness/skew is a global property of a metric
  (cross-entropy has a long right tail; accuracy compresses near a ceiling), not something that varies from one region of 
  hyperparameter space to another. A family wit 3 new messages (ctrl+End) ↓ has no way to encode a spatially-varying 
  pattern — which is precisely what makes it structurally incapable of absorbing $T$'s job, unlike the fully flexible     
  isotonic fit. And it's not a novel choice: it's what HEBO uses for output warping, for the same reason.                 
                                                                                                                          
  Now committing this to fit.py.