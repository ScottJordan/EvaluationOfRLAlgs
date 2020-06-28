function circleShape(x, y, r)
	θ = LinRange(0, 2*π, 500)
	x .+ r*sin.(θ), y .+ r*cos.(θ)
end
