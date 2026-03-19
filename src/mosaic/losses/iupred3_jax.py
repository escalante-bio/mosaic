import jax.numpy as jnp

# JAX-compatible IUPred3 implementation that maximizes disorder
def iupred3(sequence: str) -> jnp.ndarray:
    """
    Predicts the disorder of the given amino acid sequence using the IUPred3 model.

    Parameters:
    sequence (str): A string representing the amino acid sequence.

    Returns:
    jnp.ndarray: An array of disorder scores from 0 (ordered) to 1 (disordered).
    """
    # Placeholder for model loading and prediction logic
    # The actual implementation would involve loading a pre-trained IUPred3 model
    # and passing the input sequence through it.
    disorder_scores = jnp.zeros(len(sequence))  # Dummy implementation
    return disorder_scores
