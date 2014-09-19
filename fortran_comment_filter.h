
 #if !defined fortan_comment_filter_H
 #define fortan_comment_filter_H

#include <boost/iostreams/char_traits.hpp> // EOF, WOULD_BLOCK
#include <boost/iostreams/concepts.hpp>    // input_filter
#include <boost/iostreams/operations.hpp> 


namespace boost { namespace iostreams { namespace fortan_comments_input_filter_h {

class fortan_comments_input_filter : public input_filter {
public:
    explicit  fortan_comments_input_filter(char comment_char = '!')
        : comment_char_(comment_char), skip_(false)
        { }

    template<typename Source>
    int get(Source& src)
     {
        int c;
        while (true) {
            if ((c = boost::iostreams::get(src)) == EOF || c == WOULD_BLOCK)
                break;
            skip_ = c == comment_char_ ?
                true :
                c == '\n' ?
                    false :
                    skip_;
	    

	    //done add .d0 processing
            if ((!skip_)&&(c!='d')&&(c!='D'))
                break;
        }
        return c;
    }

    template<typename Source>
    void close(Source&) { skip_ = false; }
private:
    char comment_char_;
    bool skip_;
};

} } } // End namespace boost::iostreams:example

using boost::iostreams::fortan_comments_input_filter_h::fortan_comments_input_filter;

#endif
