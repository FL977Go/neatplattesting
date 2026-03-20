def generation_stats_str(stats, pop, generation):
    """
    Generate NEAT-style generation stats as a string.

    stats      : StatisticsReporter object
    pop        : Population object (live species info)
    generation : int
    Returns   : string
    """
    lines = []

    # Header
    # lines.append(f"\n ****** Running generation {generation} ******\n")

    # Population-level stats (if available)
    get_mean = getattr(stats, 'get_fitness_mean', None)
    fitness_mean_list = get_mean() if callable(get_mean) else (get_mean or [])
    get_stdev = getattr(stats, 'get_fitness_stdev', None)
    fitness_stdev_list = get_stdev() if callable(get_stdev) else (get_stdev or [])

    # Gather all genomes from current population species
    all_genomes = []
    for species in pop.species.species.values():
        all_genomes.extend(species.members.values())

    # Determine best genome (if any)
    best = None
    if all_genomes:
        # guard against genomes without fitness attribute
        best = max(all_genomes, key=lambda g: getattr(g, 'fitness', float('-inf')))

    # Find species id for best genome
    best_species_id = None
    if best is not None:
        for sid, s in pop.species.species.items():
            if best.key in s.members:
                best_species_id = sid
                break

    if generation < len(fitness_mean_list):
        mean = fitness_mean_list[generation]
        stdev = fitness_stdev_list[generation] if generation < len(fitness_stdev_list) else 0.0
        lines.append(f"Population's average fitness: {mean:.5f} stdev: {stdev:.5f}")
    else:
        lines.append("Population's average fitness: N/A")

    # Best genome
    if best is not None:
        # compute a reasonable size representation
        try:
            size = best.size()
        except Exception:
            size = (len(getattr(best, 'nodes', {})), len(getattr(best, 'connections', {})))
        lines.append(f"Best fitness: {getattr(best, 'fitness', 0.0):.5f} - size: {size} - species {best_species_id} - id {getattr(best, 'key', 'N/A')}")
    else:
        lines.append("Best genome: N/A")

    # Species-level info: compute sizes and fitness directly from the live population
    species_items = list(pop.species.species.items())
    total_members = 0
    lines.append(f"Population of {sum(len(s.members) for _, s in species_items)} members in {len(species_items)} species:")

    lines.append("   ID   age  size  fitness  adj fit  stag")
    lines.append("  ====  ===  ====  =======  =======  ====")

    for sid, species in species_items:
        age = getattr(species, "age", 0)
        size_val = len(species.members)
        # species mean fitness (guard missing fitness values)
        if size_val > 0:
            fit = sum(getattr(g, 'fitness', 0.0) for g in species.members.values()) / size_val
        else:
            fit = 0.0
        adj = getattr(species, "adjusted_fitness", 0.0) or 0.0
        stag = getattr(species, "last_improved", 0)
        lines.append(f"{sid:5d} {age:4d} {size_val:5d} {fit:8.1f} {adj:8.3f} {stag:5d}")

    # Extinction
    extinctions = getattr(stats, "complete_extinction", 0)
    lines.append(f"Total extinctions: {extinctions}")

    return "\n".join(lines)