using Documenter
using NPDemand


makedocs(
    sitename = "NPDemand",
    format = Documenter.HTML(),
    modules = [NPDemand],
    pages = [
        "index.md",
        "Implementation Details" => [
            "Setup and GMM" => "details.md",
            "Quasibayes: Priors and Sampling" => "priors.md",
            "Constraints and SMC" => "constraints.md"
        ],
        "Usage" => [
            "Basic Usage (GMM)" => "usage.md",
            "Using Quasi-Bayes Tools" => "quasibayes_usage.md"
        ],
        "Post-Estimation" => "postestimation.md",
        "Function Documentation" => "functions.md"
        # "Subsection" => [
        #     ...
        # ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
# config = Documenter.GitHubActions()
deploydocs(
    repo = "github.com/jamesbrandecon/NPDemand.jl.git",
    push_preview = true,
    devbranch = "master"
)
