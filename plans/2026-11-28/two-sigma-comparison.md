# How the 2σ comparison works

For each ELE, the network outputs a centre μ (five numbers) and a covariance Σ (a 5×5
matrix). Both live in normalised space: each parameter is put on a log10 scale and then
mapped onto −1 to 1. The diagonal of Σ gives each parameter's spread, σᵢ = √Σᵢᵢ. The
off-diagonal entries say which parameters move together.

- **Per parameter:** zᵢ = (truthᵢ − μᵢ)/σᵢ, the miss in units of the stated spread. A miss
  is |zᵢ| > 2. If the network's Gaussian were exactly right, z would follow a standard bell
  curve, and |z| > 2 would happen 4.55% of the time. That is the reference. A rate well
  above it means the error bars are too tight; well below, too loose.
- **Jointly:** r² = (truth − μ)ᵀ Σ⁻¹ (truth − μ), the Mahalanobis distance. Think of it as
  reshaping the tilted, stretched ellipsoid into a round ball and then measuring ordinary
  distance. For an exact Gaussian, r² follows a chi-square distribution with 5 degrees of
  freedom. The 2σ-equivalent cut is the radius that holds 95.45% of the mass, which is
  r ≈ 3.36, and the reference miss rate is again 4.55%.
- **Why the two disagree:** the joint test catches a truth that breaks the correlation.
  Suppose p0 and D normally rise together, but the truth has p0 high and D low. It can sit
  inside 2σ on each parameter separately and still fall far outside the ellipsoid. That is
  why the forward-model curves showed modest z values with radii of 11–21.
- **Along the principal axes:** rotating into Σ's eigenvectors gives five independent
  directions, each with its own z. This showed which directions are tight (well
  determined) and which are loose (the ridge).

## Mahalanobis distance explanation

Picture a scatter of points: say, people's heights and weights. The cloud isn't round. It's
a tilted, stretched ellipse, because tall people tend to be heavier. Now ask: how unusual is
a new person?

Ordinary distance from the centre gets this wrong. A very tall, very heavy person is far
from the centre in plain distance, but sits right along the ellipse's long axis, which is a
perfectly normal place to be. A short, heavy person can be closer in plain distance and
still be genuinely odd, because they sit off to the side where the cloud is thin. The
trouble is that plain distance treats every direction alike, while the data plainly
doesn't.

So change the space until the cloud looks round. First rotate the picture so the ellipse's
axes line up with your coordinate axes. Then squash the long axis and stretch the short one
until each has the same spread, one unit. The ellipse is now a circle. Measure ordinary
distance in this new space: that is the Mahalanobis distance. The tall-and-heavy person
barely moves from the centre. The short-and-heavy person is pushed far out, because the thin
direction was stretched.

The formula is exactly that:

$$d^2 = (x - \mu)^\top \Sigma^{-1} (x - \mu)$$

Here μ is the centre and Σ is the covariance, the matrix that describes the ellipse's tilt
and stretch. Multiplying by Σ⁻¹ is the rotate-and-rescale step written compactly. With a
single variable it reduces to the familiar z = (x − μ)/σ, distance counted in standard
deviations. Mahalanobis distance is "number of standard deviations" generalised to tilted,
correlated clouds.

Once the cloud is round, every direction is equivalent, so a "2σ region" is simply a ball.
Back in the original space it becomes an ellipsoid, the shape the network draws around its
guess. How big a ball counts as 2σ depends on the number of dimensions: the more axes there
are, the more room a point has to stray, so the ball must be bigger. In five dimensions it
has radius about 3.4, not 2.

> The one line to keep: Mahalanobis distance asks how far away you are compared with how
> spread out things are in that direction.

In our case, the network's Σ says "p0 and D move together". A truth that breaks that
pattern can look fine one parameter at a time and still be far outside the ellipsoid. That
is what the forward-model curves did.
