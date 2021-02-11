install.packages("mice")
df <- nhanes
df
#Output the percentage missingness for each feature as a vector.
Per <- colSums(is.na(df))/nrow(df)*100
Per
#Show these percentages as a barplot.

tibble(
  nmissing = colSums(is.na(df)),
  pmissing = nmissing/nrow(df)
) %>% 
  rownames_to_column() %>% 
  ggplot(aes(x = rowname, y = pmissing, label = nmissing)) +
  geom_bar(stat = "identity") +
  geom_text(nudge_y = 1/30) +
  ylim(0, 1) +
  labs(
    x = "Feature",
    y = "% missing", 
    title = "Missingness bargraph"
  ) +
  theme_minimal()

#Now display the missingness pattern per age group (1, 2, 3).


df %>%
  group_by(age) %>% 
  summarise_all(function(x) sum(is.na(x)) / n() * 100) %>% 
  round(2)
df
#How many rows are missing all the data except the age variable?

sum(rowSums(is.na(df)) == 3)

md.pattern(df)

#ASSIGMNENT
?fdgs
sum(is.na(fdgs))
86/(10030*8)*100




