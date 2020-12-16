using Documenter, NPDemand

makedocs(sitename = "NPDemand Documentation")
ENV["TRAVIS_PULL_REQUEST"] = false;
ENV["TRAVIS_BRANCH"] = "master";
myTravis = Documenter.Travis();
deploydocs(
    repo = "github.com/jamesbrandecon/NPDemand.jl.git",
    deploy_config = myTravis,
    push_preview = false
)
