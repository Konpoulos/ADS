dat <- read.csv('vet.data.csv')

#preprocessing
dat <- dat %>% 
  pivot_longer(cols = c(value2017, value2018), names_to = 'year', values_to = 'count') %>% 
  mutate(year = str_remove(year, "value")) %>% 
  pivot_wider(names_from = type, values_from = count) %>% 
  separate(col = gen_age, into = c('gender', 'age')) %>% 
  unite(col = ID, clinic, client, remove = FALSE) %>% 
  mutate(age = as.numeric(age))
 
dat

dat%>%
  ggplot()+
  geom_bar(aes(x=breed))+
  coord_flip()

ggplot(data=dat)+
  geom_bar(aes(x=food_quality))

ggplot(data = dat) +
  geom_bar(aes(x = area))

ggplot(data=dat)+
  geom_histogram(aes(x=weight))+
  facet_wrap(~breed)

ggplot(data=dat)+
  geom_boxplot(aes(x=weight))+
  facet_wrap(~breed)

 dat%>%
  filter(weight < 3 | weight > 20) %>% 
  select(ID)

dat$ID
which(dat$weight < 3)
which(dat$weight > 20)

dat%>%
  group_by(breed)%>%
  summarise(mean=mean(weight,na.rm = TRUE))

dat%>%
  ggplot(aes(x=age,y=weight))+
  geom_point()+
  geom_smooth(method ='lm')

dat%>%
  ggplot(aes(x=age,y=weight))+
  geom_point()+
  geom_smooth(method ='lm')+
  facet_grid(area~breed)

mean(dat%>%
       filter(year==2017)%>%
       select(weight)<
       dat%>%
       filter(year==2018)%>%
       select(weight),na.rm = TRUE)

ggplot(data = mpg) + 
  geom_point(mapping = aes(x = displ, y = hwy, color = class))

ggplot(data = mpg,mapping = aes(x = displ, y = hwy,)) +
  geom_point(mapping = aes(color=drv))+
  geom_smooth(mapping = aes(linetype = drv,color=drv))
