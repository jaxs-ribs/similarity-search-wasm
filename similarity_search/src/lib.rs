use kinode_process_lib::{call_init, println, Address};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::f32;

use std::time::{Duration, Instant};

wit_bindgen::generate!({
    path: "wit",
    world: "process",
});

fn create_random_vectors() -> Vec<Vec<f32>> {
    let mut rng = thread_rng();
    let between = Uniform::new(-100000.0, 100000.0);
    let vectors: Vec<Vec<f32>> = (0..16384)
        .map(|_| (0..512).map(|_| between.sample(&mut rng)).collect())
        .collect();
    vectors
}

fn dot_product(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn magnitude(vec: &Vec<f32>) -> f32 {
    vec.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    dot_product(a, b) / (magnitude(a) * magnitude(b))
}

fn similarity_search(input_vector: &Vec<f32>, vectors: &Vec<Vec<f32>>, top_k: usize) -> Vec<Vec<f32>> {
    let mut similarities: Vec<(f32, &Vec<f32>)> = vectors.iter()
        .map(|v| (cosine_similarity(input_vector, v), v))
        .collect();

    similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    similarities.iter()
        .take(top_k.min(25))
        .map(|(_, v)| (*v).clone())
        .collect()
}

fn benchmark() {
    let random_vectors = create_random_vectors();
    let input_vector = random_vectors[0].clone();
    let top_ks = [1, 2, 5, 25];

    for &top_k in top_ks.iter() {
        let mut durations = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _results = similarity_search(&input_vector, &random_vectors, top_k);
            let duration = start.elapsed();
            durations.push(duration);
        }
        let average_duration: Duration = durations.iter().sum::<Duration>() / durations.len() as u32;
        println!("Average time for top_k = {}: {} ns", top_k, average_duration.as_nanos());
    }
}

call_init!(init);
fn init(_our: Address) {
    println!("similarity search benchmark: begin");
    benchmark();
}

