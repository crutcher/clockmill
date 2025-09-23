use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The number of steps to run.
    #[arg(long, default_value = "100")]
    pub steps: usize,
}

fn main() {
    let args = Args::parse();

    println!("Args: {:#?}", args);
}
