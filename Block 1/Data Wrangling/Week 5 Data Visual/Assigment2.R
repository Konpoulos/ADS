# Get an idea of what the menu dataset looks like
head(menu)
str(menu)
library(tidyverse)
library(GGally)

menu <- read.csv("menu.csv")

#1 Question
#At the start its Character
# Transformation drinks
drink.fl <- menu %>% 
  filter(str_detect(Serving.Size, " fl oz.*")) %>% 
  mutate(Serving.Size = str_remove(Serving.Size, " fl oz.*")) %>% 
  mutate(Serving.Size = as.numeric(Serving.Size) * 29.5735)

drink.carton <- menu %>% 
  filter(str_detect(Serving.Size, "carton")) %>% 
  mutate(Serving.Size = str_extract(Serving.Size, "[0-9]{2,3}")) %>% 
  mutate(Serving.Size = as.numeric(Serving.Size))

# Transformation food
food <-  menu %>% 
  filter(str_detect(Serving.Size, "g")) %>% 
  mutate(Serving.Size = (str_extract(Serving.Size, "(?<=\\()[0-9]{2,4}"))) %>% 
  mutate(Serving.Size = as.numeric(Serving.Size))

# Add Type variable indicating whether an item is food or a drink 
menu2 <-  bind_rows(drink.fl, drink.carton, food) %>% 
  mutate(
    Type = case_when(
      as.character(Category) == 'Beverages' ~ 'Drinks',
      as.character(Category) == 'Coffee & Tea' ~ 'Drinks',
      as.character(Category) == 'Smoothies & Shakes' ~ 'Drinks',
      TRUE ~ 'Food'
    )
  )
str(menu2)
#now the service size is a numeric

#2 Question
menu2 %>%
  ggplot(aes(x=Category))+
  geom_bar()+
  coord_flip()

#3 Question
menu2 %>%
  ggplot(aes(x=Calories))+
  geom_histogram()
  

#4 Question
menu2 %>%
  ggplot(aes(x=Calories))+
  geom_density(fill="light blue")+
  facet_wrap(~Category)

#5 Question
menu2 %>%
  ggplot(aes(Category,Calories))+
  geom_boxplot()+
  coord_flip()+
  theme_minimal()

#6 Question
menu2 %>%
  filter(Category=="Chicken & Fish")%>%
  ggplot(aes(x=Item,y=Calories))+
  geom_col()+
  coord_flip()

#7 Question
menu2 %>%
  ggplot(aes(x=Serving.Size,y=Calories))+
  geom_point(alpha=0.5)

#8 Question
menu2 %>%
  filter(Type=="Drinks")%>%
  ggplot(aes(x=Serving.Size,y=Calories,colour=Category))+
  geom_point(alpha=0.5)+
  geom_smooth(method = 'lm' )
#9 Question
menu2




ggpairs(menu2[,c(1,3,4,5)], upper = list(continuous = "cor"), lower = list(continuous = "points")) +
  theme_minimal()



ggplot(data = mpg) + 
  geom_smooth(mapping = aes(x = displ, y = hwy, linetype = drv))

data(mpg)
