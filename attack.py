def run(self):
    width, height = self.target_image.size
    result_pixels = []

    pixel_count = 1
    for y in range(height):
        for x in range(width):
            print("Evolving pixel %s of %s" % (pixel_count, width * height))
            pixel_count += 1

            self.current_pixel = [x, y]

            pop = self.toolbox.population(n=10)
            hof = tools.HallOfFame(maxsize=1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)

            algorithms.eaSimple(population=pop, toolbox=self.toolbox,
                                cxpb=0.5, mutpb=0.2, ngen=50, stats=stats,
                                halloffame=hof, verbose=False)

            result_pixels.append(hof[0])

    return result_pixels