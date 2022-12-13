write.csv.default <- function(obj, filename, prefix=data.path)
{
    write.table(obj, 
                file=file.path(prefix, filename),
                sep=',',
                row.names=F,
                col.names=F)
}
